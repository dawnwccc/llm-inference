import gc
import re
import warnings
from datetime import datetime
from itertools import accumulate
from typing import Union, List, Dict, Any, Callable
import torch
from transformers import LogitsProcessor, LogitsWarper, LogitsProcessorList

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
        special_tokens = []
        special_tokens.extend(self.tokenizer.special_tokens.keys())
        special_tokens.extend(self.tokenizer.tokenizer.special_tokens.keys())
        return self._completion(
            prompts=prompts,
            batch_input_ids=input_ids,
            params=params,
            device=device,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_text_handler=self.parse_text,
            extra_logits_processor=[ChatGLMInvalidScoreLogitsProcessor()],
            content_length=self.context_length,
            stream_interval=stream_interval,
            special_tokens=special_tokens
        )

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
        special_tokens = []
        special_tokens.extend(self.tokenizer.special_tokens.keys())
        special_tokens.extend(self.tokenizer.tokenizer.special_tokens.keys())
        return self._completion(
            prompts=prompts,
            batch_input_ids=input_ids,
            params=params,
            device=device,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_text_handler=self.parse_text,
            extra_logits_processor=[ChatGLMInvalidScoreLogitsProcessor()],
            content_length=self.context_length,
            stream_interval=stream_interval,
            special_tokens=special_tokens
        )

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
                    lines[i] = "<br>" + line
        text = "".join(lines)
        return text

    def _completion(
            self,
            prompts: List[str],
            batch_input_ids: Union[torch.Tensor, torch.LongTensor],
            params: CompletionParams,
            device: str,
            attention_mask: Union[torch.Tensor, torch.LongTensor] = None,
            position_ids: Union[torch.Tensor, torch.LongTensor] = None,
            response_text_handler: Callable = None,
            logits_processor: LogitsProcessorList = None,
            extra_logits_processor: List[Union[LogitsProcessor, LogitsWarper]] = None,
            content_length: int = 8192,
            special_tokens: List[int] = None,
            stream_interval: int = -1
        ) -> TempCompletionResponse:
        # 初始化参数
        temperature = params.temperature
        repetition_penalty = params.repetition_penalty
        top_p = params.top_p
        max_new_tokens = params.max_tokens
        stop_str_list = params.stop_str
        echo = params.echo
        prompts_length = [len(p) for p in prompts]

        stop_token_ids = params.stop_token_ids
        if self.tokenizer.eos_token_id not in params.stop_token_ids:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        stop_token_ids.extend([self.tokenizer.get_command("<|user|>"), self.tokenizer.get_command("<|observation|>")])

        if special_tokens is None:
            if self.tokenizer.all_special_tokens is not None:
                special_tokens = self.tokenizer.all_special_tokens
            else:
                special_tokens = []

        # 批量任务处理
        task_total = len(batch_input_ids)
        batch_input_ids_length = [sum(attn_mask.tolist()) for attn_mask in attention_mask]
        batch_output_ids = batch_input_ids if echo else (
            torch.tensor([[] for _ in range(task_total)], dtype=torch.int32, device=device))
        batch_output = prompts if echo else [""] * task_total
        # 是否输出logprobs
        logprobs = params.logprobs
        is_logprobs = logprobs is not None
        batch_temp_logprobs = [{
            "text_offset": [],
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": []
        } for _ in range(task_total)]
        batch_final_output_ids = [None for _ in range(task_total)]
        batch_final_response_choices: List[Union[None, CompletionChoiceResponse]] = [
            None
            for _ in range(task_total)
        ]

        if logits_processor is None:
            logits_processor = default_logits_processor(temperature, repetition_penalty, top_p)
        if extra_logits_processor is not None:
            logits_processor.extend(extra_logits_processor)

        batch_input_ids = batch_input_ids[:, -(content_length - max_new_tokens - 1):]

        # [batch, tokens]
        next_tokens_ids = None
        token_index = [0] * task_total
        past_key_values = None
        # [False, False, False, ..., False]
        running_state = [max_new_tokens > 0] * task_total
        finish_reason: List[Union[str, CompletionFinishReasonEnum]] = [""] * task_total
        is_interrupted = False
        index = 0

        while (echo and is_logprobs and index == 0) or \
                (any(running_state) and not is_interrupted):
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
                    batch_temp_logprobs[i]["token_logprobs"] = batch_prompt_tokens_probs[i, :batch_input_ids_length[i],
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
                        batch_temp_logprobs[i]["top_logprobs"].append(top_logprobs)

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
                    batch_temp_logprobs[i]["token_logprobs"].append(batch_last_tokens_logprobs[i, 0].tolist())
                    top_tokens = self.tokenizer.batch_decode(batch_sample_output_ids[i, :])
                    top_tokens_logprobs = batch_top_tokens_logprobs[i].tolist()
                    top_logprobs = dict(zip(top_tokens, top_tokens_logprobs))
                    batch_temp_logprobs[i]["top_logprobs"].append(top_logprobs)
            else:
                if temperature < 1e-5 or top_p < 1e-8:
                    next_tokens_ids = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)
                else:
                    probs = torch.softmax(last_token_logits, dim=-1)
                    next_tokens_ids = torch.multinomial(probs, num_samples=1)

            # 判断新生成的 token 是否在 stop_token 中，若存在，则停止该任务
            for i in range(task_total):
                if next_tokens_ids[i].tolist() in stop_token_ids:
                    # 停止生成
                    running_state[i] = False
                    finish_reason[i] = CompletionFinishReasonEnum.stop
                    batch_final_output_ids[i] = batch_output_ids[i].tolist()
                else:
                    token_index[i] += 1

            # 判断当前补全词的数量是否超出最大值
            if index >= max_new_tokens:
                # 超出，则停止全部的任务
                for i in range(task_total):
                    if running_state[i]:
                        running_state[i] = False
                        finish_reason[i] = CompletionFinishReasonEnum.length
                        batch_final_output_ids[i] = batch_output_ids[i].tolist()
            else:
                batch_output_ids = torch.cat((batch_output_ids, next_tokens_ids), dim=-1)

            if index % stream_interval == 0 or not all(running_state):
                for i in range(task_total):
                    if not running_state[i] and batch_final_response_choices[i] is not None:
                        # 若已经结束且已经构造了 response，则跳过
                        continue
                    if running_state[i]:
                        # 任务仍在运行
                        output_str = self.tokenizer.decode(batch_output_ids[i])
                        # 根据stop_str判断是否需要停止
                        is_stop, output_str = check_stop_str(
                            output_str, stop_str_list, check_start=prompts_length[i] if echo else 0)
                        if is_stop:
                            running_state[i] = False
                            finish_reason[i] = CompletionFinishReasonEnum.stop
                        batch_output[i] = output_str
                    else:
                        output_str = self.tokenizer.decode(batch_final_output_ids[i])

                    if not running_state[i] and batch_final_response_choices[i] is None:
                        # 若已经结束但没有构建response，则构建
                        if is_logprobs:
                            # 构建logprobs
                            output_tokens_str = self.tokenizer.batch_decode(batch_final_output_ids[i])
                            print(special_tokens)
                            for s in output_tokens_str:
                                print(f"{s}: {s in special_tokens}")
                            output_tokens_flag = [
                                not (s in special_tokens) and s
                                for s in output_tokens_str
                            ]
                            # if echo:
                            #     # 去除第一个token，即prompt中的第一个token
                            #     output_tokens_flag = output_tokens_flag[1:]
                            output_tokens_str = [
                                token_str
                                for flag, token_str in zip(output_tokens_flag, output_tokens_str)
                                if flag
                            ]
                            # 计算logprobs: text_offset
                            output_tokens_str_length = [len(token_str) for flag, token_str in
                                                        zip(output_tokens_flag, output_tokens_str) if flag]
                            batch_temp_logprobs[i]["tokens"] = output_tokens_str
                            batch_temp_logprobs[i]["text_offset"] = [0] + list(accumulate(output_tokens_str_length))[:-1]
                            batch_temp_logprobs[i]["token_logprobs"] = [
                                tok_logprobs
                                for flag, tok_logprobs in zip(output_tokens_flag[1:], batch_temp_logprobs[i]["token_logprobs"][1:])
                                if flag
                            ]
                            batch_temp_logprobs[i]["top_logprobs"] = [
                                top_logprobs
                                for flag, top_logprobs in zip(output_tokens_flag[1:], batch_temp_logprobs[i]["top_logprobs"][1:])
                                if flag
                            ]
                        batch_final_response_choices[i] = CompletionChoiceResponse(
                            end=int(datetime.now().timestamp() * 1000),
                            text=output_str,
                            finish_reason=finish_reason[i],
                            logprobs=CompletionLogprobs(**batch_temp_logprobs[i]) if is_logprobs else None,
                            usage=CompletionUsageInfo(
                                prompt_tokens=batch_input_ids_length[i],
                                completion_tokens=token_index[i],
                                total_tokens=batch_input_ids_length[i] + token_index[i]
                            )
                        )

                response = TempCompletionResponse(
                    choices=[
                        CompletionChoiceResponse(
                            text=batch_output[i] if response_text_handler is None else response_text_handler(batch_output[i]),
                            logprobs=None,
                            usage=CompletionUsageInfo(
                                prompt_tokens=batch_input_ids_length[i],
                                completion_tokens=token_index[i],
                                total_tokens=batch_input_ids_length[i] + token_index[i]
                            ),
                            finish_reason=None
                        ) if batch_final_response_choices[i] is None else batch_final_response_choices[i]
                        for i in range(task_total)
                    ],
                    interrupted=is_interrupted
                )
                yield response
                is_interrupted = response.interrupted
            index += 1
            new_mask = torch.tensor(running_state, dtype=torch.int8).unsqueeze(1).to(device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            if position_ids is not None:
                position_ids = position_ids[:, -1:].clone() + 1

        del past_key_values, out, input_ids, attention_mask, position_ids
        gc.collect()
        torch.cuda.empty_cache()

    def embedding(self):
        pass
