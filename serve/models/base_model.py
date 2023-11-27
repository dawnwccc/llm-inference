import gc
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
from itertools import accumulate
from serve.entity.exception import GlobalException
from serve.utils.inference.batcher import batch_tokenize
from serve.utils.inference.logits_processor import default_logits_processor
from serve.utils.inference.stopping_criteria import check_stop_str, default_stopping_criteria
import torch
from serve.utils.enums import CompletionFinishReasonEnum
from serve.utils.chat_template import ChatTemplate
from serve.utils.factory import register_model_function

from serve.entity.inference import TempCompletionResponse, CompletionParams
from serve.entity.protocol import CompletionChoiceResponse, CompletionLogprobs, CompletionUsageInfo, ChatMessage


class AbstractModelFunction:
    """模型功能抽象类"""

    def __init__(self, tokenizer, model, device, context_length):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.context_length = context_length
        # 检查模型是否含有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({
                "pad_token": "<PAD>"
            })
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def stream_completion(
            self, prompt: Union[str, List[str]], params: CompletionParams, stream_interval: int = 2
    ) -> TempCompletionResponse:
        """流式文本补全"""
        # TODO: 判断设备使用情况
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

    @abstractmethod
    def _completion(
            self,
            prompts: List[str],
            batch_input_ids: Union[torch.Tensor, torch.LongTensor],
            params: CompletionParams,
            device: str,
            attention_mask: Union[torch.Tensor, torch.LongTensor] = None,
            position_ids: Union[torch.Tensor, torch.LongTensor] = None,
            stream_interval: int = -1
    ) -> TempCompletionResponse:
        """文本补全"""
        raise NotImplementedError("未实现的 completion 方法")

    def stream_chat_completion(
            self, chat_template: ChatTemplate, messages: List[ChatMessage], params: CompletionParams,
            stream_interval: int = 2
    ) -> TempCompletionResponse:
        prompt = chat_template.complete_message(messages)
        # TODO: 判断设备使用情况
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

    @abstractmethod
    def embedding(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


# @register_model_function("default")
class DefaultModelFunction(AbstractModelFunction):

    def __init__(self, tokenizer, model, device, context_length):
        super().__init__(tokenizer, model, device, context_length)

    @torch.inference_mode()
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
        max_length = batch_input_ids.shape[1] + max_new_tokens
        # 是否打印提示词
        echo = params.echo
        stop_token_ids = [torch.as_tensor([token_ids], device=device) for token_ids in params.stop_token_ids]
        if self.tokenizer.eos_token_id not in params.stop_token_ids:
            stop_token_ids.append(torch.as_tensor([self.tokenizer.eos_token_id], device=device))
        for stop_str in stop_str_list:
            stop_token_ids.append(self.tokenizer(stop_str, return_tensors="pt").input_ids[0].to(device))
        # 批量任务处理
        task_total = len(batch_input_ids)
        batch_input_ids_length = [sum(attn_mask.tolist()) for attn_mask in attention_mask]
        batch_output_ids = [input_ids for input_ids in batch_input_ids] if echo else \
            [torch.tensor([], dtype=torch.int32) for _ in range(task_total)]
        batch_output = prompts if echo else [""] * task_total
        # 是否输出logprobs
        logprobs = params.logprobs
        is_logprobs = logprobs is not None
        # [batch, tokens, token_ids]
        batch_sample_output_ids = []
        batch_logprobs = [{
            "text_offset": [],
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": []
        } for _ in range(task_total)]

        # construct logits processor
        logits_processor = default_logits_processor(temperature, repetition_penalty, top_p)
        # construct stopping criteria
        stopping_criteria = default_stopping_criteria(max_length, stream_interval, 60 * 5,
                                                      stop_token_ids,
                                                      start_index=0 if echo else batch_input_ids.shape[1])

        if self.model.config.is_encoder_decoder:
            batch_input_ids = batch_input_ids[:, -self.context_length:]
        else:
            batch_input_ids = batch_input_ids[:, -(self.context_length - max_new_tokens - 1):]

        encoder_output = None
        if self.model.config.is_encoder_decoder:
            encoder_output = self.model.encoder(
                input_ids=batch_input_ids
            )[0]
            start_ids = torch.as_tensor(
                [[self.model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=device,
            )
        else:
            start_ids = batch_input_ids

        # [batch, tokens]
        next_tokens_ids = None
        token_index = [0] * task_total
        past_key_values = None
        running_state = [True] * task_total
        finish_reason = [""] * task_total
        is_interrupted = False
        index = 0

        while any(running_state) and not is_interrupted:
            if index == 0:
                input_ids = start_ids
            else:
                input_ids = next_tokens_ids
            if self.model.config.is_encoder_decoder:
                out = self.model.decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                logits = self.model.lm_head(out[0])
            else:
                out = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values
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
                batch_raw_logprobs = torch.log_softmax(logits, dim=-1)
                _, batch_token_ids = torch.topk(batch_raw_logprobs, logprobs, dim=-1)
                # 第一个token没有logprobs
                batch_sample_output_ids = batch_token_ids[:, 1:, :]
                batch_prompt_tokens_probs = torch.gather(
                    batch_raw_logprobs[:, :-1, :], dim=-1, index=batch_input_ids.unsqueeze(2)[:, 1:, :]
                )
                batch_top_tokens_logprobs = torch.gather(
                    batch_raw_logprobs[:, :-1, :], dim=-1,
                    index=batch_sample_output_ids
                )
                for i in range(task_total):
                    batch_logprobs[i]["token_logprobs"] = batch_prompt_tokens_probs[i, :batch_input_ids_length[i],
                                                          0].tolist()
                    top_tokens = [
                        self.tokenizer.batch_decode(batch_sample_output_ids[i, j, :])
                        for j in range(batch_input_ids_length[i] - 1)
                    ]
                    top_tokens_logprobs = batch_top_tokens_logprobs[i, :batch_input_ids_length[i], :].tolist()
                    for toks, toks_logprobs in zip(top_tokens, top_tokens_logprobs):
                        top_logprobs = dict(zip(toks, toks_logprobs))
                        batch_logprobs[i]["top_logprobs"].append(top_logprobs)

            if logits_processor:
                last_token_logits = [
                    logits_processor(
                        torch.as_tensor(batch_output_ids[i], device=logits.device).long(),
                        raw_last_logits[i, :])
                    for i in range(task_total)
                ]
                last_token_logits = torch.stack(last_token_logits).to(device)
            else:
                last_token_logits = raw_last_logits

            # next_token_ids = None
            if is_logprobs:
                if temperature < 1e-5 or top_p < 1e-8:
                    _, batch_tokens_ids = torch.topk(last_token_logits, logprobs, dim=-1)
                    next_tokens_ids = batch_tokens_ids[:, 0:1]
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

            index += 1
            for i in range(task_total):
                if not running_state[i]:
                    continue
                token_index[i] += 1
                batch_output_ids[i] = torch.cat((batch_output_ids[i], next_tokens_ids[i, :]), dim=-1)

            if index % stream_interval == 0 or not all(running_state):
                for i in range(task_total):
                    if not running_state[i]:
                        continue
                    message = stopping_criteria(batch_output_ids[i], last_token_logits)
                    # 必须放在 stopping_criteria 之后，因为在该方法中会修改output_ids
                    output_tokens_str = self.tokenizer.batch_decode(
                        batch_output_ids[i],
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
                    output_tokens_str = list(filter(None, output_tokens_str))
                    batch_output[i] = "".join(output_tokens_str)
                    if message.stop:
                        running_state[i] = False
                        finish_reason[i] = message.message
                        # 计算logprobs: text_offset
                        output_tokens_str_length = [len(token_str) for token_str in output_tokens_str]
                        batch_logprobs[i]["tokens"] = output_tokens_str
                        batch_logprobs[i]["text_offset"] = [0] + list(accumulate(output_tokens_str_length))[:-1]
                        batch_logprobs[i]["token_logprobs"] = batch_logprobs[i]["token_logprobs"][
                                                              :len(output_tokens_str)]
                        batch_logprobs[i]["top_logprobs"] = batch_logprobs[i]["top_logprobs"][:len(output_tokens_str)]

                response = TempCompletionResponse(
                    choices=[
                        CompletionChoiceResponse(
                            text=batch_output[i],
                            logprobs=None if running_state[i] else CompletionLogprobs(**batch_logprobs[i]),
                            usage=CompletionUsageInfo(
                                prompt_tokens=batch_input_ids_length[i],
                                completion_tokens=token_index[i],
                                total_tokens=batch_input_ids_length[i] + token_index[i]
                            ),
                            finish_reason=None if running_state[i] else finish_reason[i]
                        )
                        for i in range(task_total)
                    ],
                    interrupted=False
                )
                yield response
                is_interrupted = response.interrupted
            next_tokens_ids.to(device)
            new_mask = torch.tensor(running_state, dtype=torch.int8).unsqueeze(1).to(device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        del past_key_values, out, input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def embedding(self):
        pass
