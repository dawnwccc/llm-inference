import gc
import re
import warnings
from itertools import accumulate
from typing import Union, List, Dict, Any
import torch

from serve.entity.exception import GlobalException
from serve.entity.inference import TempCompletionResponse, CompletionParams
from serve.entity.protocol import CompletionChoiceResponse, CompletionUsageInfo, ChatMessage, CompletionLogprobs
from serve.utils.chat_template import ChatTemplate
from serve.utils.factory import register_model_function
from serve.utils.inference.batcher import batch_tokenize
from serve.utils.inference.logits_processor import default_logits_processor
from serve.utils.inference.stopping_criteria import check_stop_str, default_stopping_criteria
from serve.models.base_model import AbstractModelFunction
from serve.utils.inference.logits_processor import ChatGLMInvalidScoreLogitsProcessor
from serve.utils.enums import CompletionFinishReasonEnum


def chatglm_process_output(response):
    # response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response


# @register_model_function("chatglm3")
class ChatGLM3ModelFunction(AbstractModelFunction):

    def __init__(self, tokenizer, model, device, context_length):
        super().__init__(tokenizer, model, device, context_length)

    def stream_completion(self, prompt: Union[str, List[str]], params: CompletionParams,
                          stream_interval: int = 2) -> TempCompletionResponse:
        device = self.device
        n = params.n
        prompts, (input_ids, attention_mask, position_ids) = batch_tokenize(
            prompt=prompt, n=n, device=device,
            tokenize_func=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id

        )
        return self._completion(
            prompts=prompts,
            batch_input_ids=input_ids,
            params=params,
            device=device,
            attention_mask=attention_mask,
            position_ids=position_ids,
            stream_interval=stream_interval
        )

    def _completion(self, prompts: List[str], batch_input_ids: Union[torch.Tensor, torch.LongTensor],
                    params: CompletionParams, device: str, attention_mask: Union[torch.Tensor, torch.LongTensor] = None,
                    position_ids: Union[torch.Tensor, torch.LongTensor] = None,
                    stream_interval: int = -1) -> TempCompletionResponse:
        # 初始化参数
        temperature = params.temperature
        repetition_penalty = params.repetition_penalty
        top_p = params.top_p
        max_new_tokens = params.max_tokens
        stop_str_list = params.stop_str
        # 是否打印提示词
        echo = params.echo
        max_length = batch_input_ids.shape[1] + max_new_tokens if echo else max_new_tokens
        stop_token_ids = [torch.as_tensor([token_ids], device=device) for token_ids in params.stop_token_ids]
        if self.tokenizer.eos_token_id not in params.stop_token_ids:
            stop_token_ids.append(torch.as_tensor([self.tokenizer.eos_token_id], device=device))
        if self.tokenizer.get_command("<|user|>") not in params.stop_token_ids:
            stop_token_ids.append(torch.as_tensor([self.tokenizer.get_command("<|user|>")], device=device))
        for stop_str in stop_str_list:
            stop_token_ids.append(self.tokenizer(stop_str, return_tensors="pt").input_ids[0].to(device))
        # 批量任务处理
        task_total = len(batch_input_ids)
        batch_input_ids_length = [sum(attn_mask.tolist()) for attn_mask in attention_mask]
        batch_output_ids = batch_input_ids if echo else (
            torch.tensor([[] for _ in range(task_total)], dtype=torch.int32, device=device))
        batch_output = prompts if echo else [""] * task_total
        # 是否输出logprobs
        logprobs = params.logprobs
        is_logprobs = logprobs is not None
        # [batch, tokens, token_ids]
        batch_sample_output_ids = torch.tensor([], dtype=torch.int32)
        batch_logprobs = [{
            "text_offset": [],
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": []
        } for _ in range(task_total)]

        # construct logits processor
        logits_processor = default_logits_processor(temperature, repetition_penalty, top_p)
        logits_processor.append(ChatGLMInvalidScoreLogitsProcessor())
        # construct stopping criteria
        stopping_criteria = default_stopping_criteria(max_length, stream_interval, 60 * 5,
                                                      stop_token_ids,
                                                      start_index=0 if echo else batch_input_ids.shape[1])

        batch_input_ids = batch_input_ids[:, -(self.context_length - max_new_tokens - 1):]

        # [batch, tokens]
        next_tokens_ids = None
        token_index = [0] * task_total
        past_key_values = out = input_ids = None
        running_state = [max_new_tokens > 0] * task_total
        finish_reason = [""] * task_total
        is_interrupted = False
        index = 0

        while (echo and is_logprobs and index == 0) or (any(running_state) and not is_interrupted):
            if index == 0:
                input_ids = batch_input_ids
            else:
                input_ids = next_tokens_ids
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = out.logits
            if index == 0:
                raw_last_logits = torch.stack([
                    logits[i, last_valid_i - 1, :]
                    for i, last_valid_i in enumerate(batch_input_ids_length)
                ]).to(logits.device)
            else:
                raw_last_logits = logits[:, -1, :]
            past_key_values = out.past_key_values
            # 计算prompt的logprobs
            if echo and is_logprobs and index == 0:
                batch_raw_logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                batch_top_tokens_logprobs, batch_sample_output_ids = torch.topk(batch_raw_logprobs, logprobs, dim=-1)
                # 第一个token没有logprobs
                batch_prompt_tokens_probs = torch.gather(
                    batch_raw_logprobs, dim=-1, index=batch_input_ids.unsqueeze(2)[:, 1:, :]
                )
                for i in range(task_total):
                    batch_logprobs[i]["token_logprobs"] = batch_prompt_tokens_probs[i, :batch_input_ids_length[i],
                                                          0].tolist()
                    # 去掉第一个词和最后一个词
                    top_tokens = [
                        self.tokenizer.batch_decode(batch_sample_output_ids[i, j, :])
                        # 去掉第一个和最后一个
                        for j in range(batch_input_ids_length[i] - 1)
                    ]
                    top_tokens_logprobs = batch_top_tokens_logprobs[i, :batch_input_ids_length[i] - 1, :].tolist()
                    for toks, toks_logprobs in zip(top_tokens, top_tokens_logprobs):
                        top_logprobs = dict(zip(toks, toks_logprobs))
                        batch_logprobs[i]["top_logprobs"].append(top_logprobs)

            if logits_processor:
                last_token_logits = logits_processor(
                        torch.as_tensor(batch_output_ids, device=logits.device).long(),
                        raw_last_logits
                )
            else:
                last_token_logits = raw_last_logits

            # next_token_ids = None
            if is_logprobs:
                if temperature < 1e-5 or top_p < 1e-8:
                    _, batch_sample_output_ids = torch.topk(last_token_logits, logprobs, dim=-1)
                    next_tokens_ids = batch_sample_output_ids[:, 0:1]
                else:
                    # 随机采样, 取出采样结果中概率最大的 token
                    probs = torch.softmax(last_token_logits, dim=-1)
                    batch_sample_output_ids = torch.multinomial(probs, num_samples=logprobs)
                    batch_sample_tokens_logprobs = torch.gather(probs, dim=-1, index=batch_sample_output_ids)
                    _, batch_tokens_ids_index = torch.topk(batch_sample_tokens_logprobs, 1, dim=-1)
                    next_tokens_ids = torch.gather(batch_sample_output_ids, dim=-1, index=batch_tokens_ids_index)
                batch_raw_last_logprobs = torch.log_softmax(raw_last_logits, dim=-1)
                batch_last_tokens_logprobs = torch.gather(batch_raw_last_logprobs, dim=-1, index=next_tokens_ids)
                batch_top_tokens_logprobs = torch.gather(batch_raw_last_logprobs, dim=-1, index=batch_sample_output_ids)
                for i in range(task_total):
                    batch_logprobs[i]["token_logprobs"].append(batch_last_tokens_logprobs[i, 0].tolist())
                    top_tokens = self.tokenizer.batch_decode(batch_sample_output_ids[i, :])
                    top_tokens_logprobs = batch_top_tokens_logprobs[i].tolist()
                    top_logprobs = dict(zip(top_tokens, top_tokens_logprobs))
                    batch_logprobs[i]["top_logprobs"].append(top_logprobs)
            else:
                if temperature < 1e-5 or top_p < 1e-8:
                    next_tokens_ids = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)
                else:
                    probs = torch.softmax(last_token_logits, dim=-1)
                    next_tokens_ids = torch.multinomial(probs, num_samples=1)

            batch_output_ids = torch.cat((batch_output_ids, next_tokens_ids), dim=-1)
            for i in range(task_total):
                if not running_state[i]:
                    continue
                token_index[i] += 1
            next_tokens_ids.to(device)
            new_mask = torch.tensor(running_state, dtype=torch.int8).unsqueeze(1).to(device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            # attention_mask = new_mask
            if position_ids is not None:
                position_ids = position_ids[:, -1:].clone() + 1

            if index % stream_interval == 0 or not all(running_state):
                for i in range(task_total):
                    if not running_state[i] and index != 0:
                        continue
                    message = stopping_criteria(batch_output_ids[i], last_token_logits,
                                                attention_mask=attention_mask[i])
                    print(batch_output_ids[i])
                    output_ids = torch.masked_select(batch_output_ids[i], attention_mask[i].bool())
                    print(output_ids)
                    output_tokens_str = self.tokenizer.batch_decode(output_ids)
                    output_tokens_flag = [
                        not (s in self.tokenizer.tokenizer.special_tokens)
                        for s in output_tokens_str
                    ]
                    output_tokens_str = [
                        token_str
                        for flag, token_str in zip(output_tokens_flag, output_tokens_str)
                        if flag
                    ]
                    # batch_output[i] = "".join(output_tokens_str)
                    batch_output[i] = self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
                    if message.stop:
                        running_state[i] = False
                        finish_reason[i] = message.message
                        if is_logprobs:
                            if echo:
                                output_tokens_flag = output_tokens_flag[1:]
                            # 计算logprobs: text_offset
                            output_tokens_str_length = [len(token_str) for flag, token_str in
                                                        zip(output_tokens_flag, output_tokens_str) if flag]
                            batch_logprobs[i]["tokens"] = output_tokens_str
                            batch_logprobs[i]["text_offset"] = [0] + list(accumulate(output_tokens_str_length))[:-1]
                            batch_logprobs[i]["token_logprobs"] = [
                                tok_logprobs
                                for flag, tok_logprobs in zip(output_tokens_flag, batch_logprobs[i]["token_logprobs"])
                                if flag
                            ]
                            batch_logprobs[i]["top_logprobs"] = [
                                top_logprobs
                                for flag, top_logprobs in zip(output_tokens_flag, batch_logprobs[i]["top_logprobs"])
                                if flag
                            ]
                response = TempCompletionResponse(
                    choices=[
                        CompletionChoiceResponse(
                            text=self.parse_text(batch_output[i]),
                            logprobs=CompletionLogprobs(**batch_logprobs[i]) if not running_state[i] and is_logprobs else None,
                            usage=CompletionUsageInfo(
                                prompt_tokens=batch_input_ids_length[i],
                                completion_tokens=token_index[i],
                                total_tokens=batch_input_ids_length[i] + token_index[i]
                            ),
                            finish_reason=None if running_state[i] else finish_reason[i]
                        )
                        for i in range(task_total)
                    ],
                    interrupted=is_interrupted
                )
                yield response
                is_interrupted = response.interrupted

            index += 1

        del past_key_values, out, input_ids, attention_mask, position_ids
        gc.collect()
        torch.cuda.empty_cache()

    def stream_chat_completion(
            self, chat_template: ChatTemplate, messages: List[ChatMessage], params: CompletionParams,
            stream_interval: int = 2
    ) -> TempCompletionResponse:
        n = params.n
        device = self.device
        conversation = chat_template.parse(messages)
        history = []
        history.extend(conversation.system_messages)
        history.extend(conversation.few_show_messages)
        history.extend(conversation.history_messages)
        prompt = conversation.prompt
        prompts, (input_ids, attention_mask, position_ids) = batch_tokenize(
            prompt=prompt, n=n, device=device,
            tokenize_func=self.tokenizer.build_chat_input, tokenize_func_kwargs={
                "history": history,
                "role": "user"
            },
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self._completion(
            prompts=prompts,
            batch_input_ids=input_ids,
            params=params,
            device=device,
            attention_mask=attention_mask,
            position_ids=position_ids,
            stream_interval=stream_interval
        )

    def embedding(self):
        pass

    def parse_text(self, text):
        """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
        lines = text.split("\n")
        lines = [line for line in lines if line != ""]
        count = 0
        for i, line in enumerate(lines):
            if "```" in line:
                count += 1
                items = line.split('`')
                if count % 2 == 1:
                    lines[i] = f'<pre><code class="language-{items[-1]}">'
                else:
                    lines[i] = f'<br></code></pre>'
            else:
                if i > 0:
                    if count % 2 == 1:
                        line = line.replace("`", "\`")
                        line = line.replace("<", "&lt;")
                        line = line.replace(">", "&gt;")
                        line = line.replace(" ", "&nbsp;")
                        line = line.replace("*", "&ast;")
                        line = line.replace("_", "&lowbar;")
                        line = line.replace("-", "&#45;")
                        line = line.replace(".", "&#46;")
                        line = line.replace("!", "&#33;")
                        line = line.replace("(", "&#40;")
                        line = line.replace(")", "&#41;")
                        line = line.replace("$", "&#36;")
                    lines[i] = "<br>"+line
        text = "".join(lines)
        return text
