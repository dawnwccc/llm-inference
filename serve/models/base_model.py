import gc
from abc import abstractmethod
from datetime import datetime
from typing import Union, List, Callable
from itertools import accumulate

from transformers import LogitsProcessorList, LogitsProcessor, LogitsWarper

from serve.entity.exception import GlobalException
from serve.utils.inference.batcher import batch_tokenize
from serve.utils.inference.logits_processor import default_logits_processor
from serve.utils.inference.stopping_criteria import check_stop_str
import torch
from serve.utils.enums import CompletionFinishReasonEnum
from serve.utils.chat_template import ChatTemplate

from serve.entity.inference import TempCompletionResponse, CompletionParams
from serve.entity.protocol import CompletionChoiceResponse, CompletionLogprobs, CompletionUsageInfo, ChatMessage


class AbstractModelFunction:
    """模型功能抽象类"""

    @abstractmethod
    def stream_completion(
            self, prompt: Union[str, List[str]], params: CompletionParams, stream_interval: int = 2
    ) -> TempCompletionResponse:
        """流式文本补全"""
        pass

    @abstractmethod
    def stream_chat_completion(
            self, chat_template: ChatTemplate, messages: List[ChatMessage], params: CompletionParams,
            stream_interval: int = 2
    ) -> TempCompletionResponse:
        """流式对话补全"""
        pass

    @abstractmethod
    def embedding(self):
        pass

    @abstractmethod
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
            stream_interval: int = -1,
            **kwargs
    ) -> TempCompletionResponse:
        """文本补全"""
        pass


class DefaultModelFunction(AbstractModelFunction):
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
            content_length=self.context_length,
            special_tokens=self.tokenizer.all_special_tokens,
            stream_interval=stream_interval
        )

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
            content_length=self.context_length,
            special_tokens=self.tokenizer.all_special_tokens,
            stream_interval=stream_interval
        )

    @torch.inference_mode()
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
            stream_interval: int = -1,
            **kwargs
    ) -> TempCompletionResponse:
        """文本补全"""
        # 初始化参数
        temperature = params.temperature
        repetition_penalty = params.repetition_penalty
        top_p = params.top_p
        max_new_tokens = params.max_tokens
        stop_str_list = params.stop_str
        echo = params.echo
        # 记录prompt的长度，用于记录在echo=True的情况下，检查stop_str时的起始位置
        prompts_length = [len(p) for p in prompts]
        # 停止token_id
        stop_token_ids = params.stop_token_ids
        if self.tokenizer.eos_token_id not in params.stop_token_ids:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        # tokenizer中的特殊token，用于避免显示在logprobs中
        if special_tokens is None:
            if self.tokenizer.all_special_tokens is not None:
                special_tokens = self.tokenizer.all_special_tokens
            else:
                special_tokens = []

        # 批量任务处理
        task_total = len(batch_input_ids)
        # 记录每个任务的input_ids的长度，用于usage.prompt_tokens
        # 由于input_ids中有pad的原因，所以需要attention_mask来计算
        batch_input_ids_length = [sum(attn_mask.tolist()) for attn_mask in attention_mask]
        # 记录每个任务的全部输入输出，用于输入到模型中
        batch_output_ids = batch_input_ids.clone().to(device) if echo else (
            torch.tensor([[] for _ in range(task_total)], dtype=torch.int32, device=device))
        init_batch_output_ids_length = [len(ids) for ids in batch_output_ids]
        # 用于记录每个任务的输出文本
        batch_output = prompts if echo else [""] * task_total
        # 是否输出logprobs
        logprobs = params.logprobs
        is_logprobs = logprobs is not None
        batch_logprobs = [{
            "text_offset": [],
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": []
        } for _ in range(task_total)]

        if logits_processor is None:
            logits_processor = default_logits_processor(temperature, repetition_penalty, top_p)
        if extra_logits_processor is not None:
            logits_processor.extend(extra_logits_processor)
        # 截取前面的context_length个token
        batch_input_ids = batch_input_ids[:, -(content_length - max_new_tokens - 1):]

        # [batch, tokens]
        encoder_output = None
        # 记录每个任务生成了多少token，用于usage.completion_tokens
        token_index = [0] * task_total
        past_key_values = out = None
        # [True, True, True, ..., True]，True表示未结束，False表示已结束
        running_state = [max_new_tokens > 0] * task_total
        # 记录任务结束原因
        finish_reason: List[Union[str, CompletionFinishReasonEnum]] = [""] * task_total
        is_interrupted = False
        # 记录共推理多少次
        index = 0

        while any(running_state) and not is_interrupted:
            if self.model.config.is_encoder_decoder:
                if index == 0:
                    if logprobs is not None:
                        raise GlobalException("暂不支持logprobs模式下的encoder-decoder模型")
                    encoder_output = self.model.encoder(
                        input_ids=torch.as_tensor(batch_input_ids, device=device)
                    )
                    batch_input_ids = torch.as_tensor(
                        [[self.model.generation_config.decoder_start_token_id] for _ in range(task_total)],
                        dtype=torch.int32,
                        device=device,
                    )
                out = self.model.decoder(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = self.model.lm_head(out[0])
            else:
                out = self.model(
                    input_ids=batch_input_ids,
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
            if index == 0 and echo and is_logprobs:
                batch_raw_logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                batch_top_tokens_logprobs, batch_sample_output_ids = torch.topk(batch_raw_logprobs, logprobs,
                                                                                dim=-1)
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
                    batch_input_ids = batch_sample_output_ids[:, 0:1]
                else:
                    # 随机采样, 取出采样结果中概率最大的 token
                    probs = torch.softmax(last_token_logits, dim=-1)
                    batch_sample_output_ids = torch.multinomial(probs, num_samples=logprobs)
                    batch_sample_tokens_logprobs = torch.gather(probs, dim=-1, index=batch_sample_output_ids)
                    _, batch_tokens_ids_index = torch.topk(batch_sample_tokens_logprobs, 1, dim=-1)
                    batch_input_ids = torch.gather(batch_sample_output_ids, dim=-1, index=batch_tokens_ids_index)
                batch_raw_last_logprobs = torch.log_softmax(raw_last_logits, dim=-1)
                batch_last_tokens_logprobs = torch.gather(batch_raw_last_logprobs, dim=-1, index=batch_input_ids)
                batch_top_tokens_logprobs = torch.gather(batch_raw_last_logprobs, dim=-1,
                                                         index=batch_sample_output_ids)
                for i in range(task_total):
                    batch_logprobs[i]["token_logprobs"].append(batch_last_tokens_logprobs[i, 0].tolist())
                    top_tokens = self.tokenizer.batch_decode(batch_sample_output_ids[i, :])
                    top_tokens_logprobs = batch_top_tokens_logprobs[i].tolist()
                    top_logprobs = dict(zip(top_tokens, top_tokens_logprobs))
                    batch_logprobs[i]["top_logprobs"].append(top_logprobs)
            else:
                if temperature < 1e-5 or top_p < 1e-8:
                    batch_input_ids = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)
                else:
                    probs = torch.softmax(last_token_logits, dim=-1)
                    batch_input_ids = torch.multinomial(probs, num_samples=1)

            # 判断当前补全词的数量是否超出最大值
            if index >= max_new_tokens:
                # 超出，则停止全部的任务
                for i in range(task_total):
                    if running_state[i]:
                        running_state[i] = False
                        finish_reason[i] = CompletionFinishReasonEnum.length

            # 判断新生成的 token 是否在 stop_token 中，若存在，则停止该任务
            for i in range(task_total):
                if not running_state[i]:
                    continue
                if batch_input_ids[i].tolist() in stop_token_ids:
                    # 停止生成
                    running_state[i] = False
                    finish_reason[i] = CompletionFinishReasonEnum.stop
                else:
                    token_index[i] += 1

            batch_output_ids = torch.cat((batch_output_ids, batch_input_ids), dim=-1)

            if index % stream_interval == 0 or not all(running_state):
                for i in range(task_total):
                    if not running_state[i] and len(batch_output[i]) > 0:
                        # 若已经结束且已经生成则跳过
                        continue
                    batch_output[i] = self.tokenizer.decode(
                        batch_output_ids[i][:token_index[i]+init_batch_output_ids_length[i]],
                        skip_special_tokens=True
                    )
                    # 根据stop_str判断是否需要停止
                    is_stop, batch_output[i] = check_stop_str(
                        batch_output[i], stop_str_list, check_start=prompts_length[i] if echo else 0)
                    if is_stop:
                        running_state[i] = False
                        finish_reason[i] = CompletionFinishReasonEnum.stop
                    if not running_state[i] and is_logprobs:
                        # 若已经任务已经结束但没有构建构建output_str和logprobs
                        output_tokens_str = self.tokenizer.batch_decode(batch_output_ids[i][:token_index[i]+init_batch_output_ids_length[i]])
                        output_tokens_flag = [
                            not (s in special_tokens) and s
                            for s in output_tokens_str
                        ]
                        output_tokens_str = [
                            token_str
                            for flag, token_str in zip(output_tokens_flag, output_tokens_str)
                            if flag
                        ]
                        output_tokens_str_length = [len(token_str) for flag, token_str in
                                                    zip(output_tokens_flag, output_tokens_str) if flag]
                        batch_logprobs[i]["tokens"] = output_tokens_str
                        batch_logprobs[i]["text_offset"] = [0] + list(accumulate(output_tokens_str_length))[:-1]
                        batch_logprobs[i]["token_logprobs"] = [
                            tok_logprobs
                            for flag, tok_logprobs in
                            zip(output_tokens_flag[1:], batch_logprobs[i]["token_logprobs"][1:])
                            if flag
                        ]
                        batch_logprobs[i]["top_logprobs"] = [
                            top_logprobs
                            for flag, top_logprobs in
                            zip(output_tokens_flag[1:], batch_logprobs[i]["top_logprobs"][1:])
                            if flag
                        ]
                response = TempCompletionResponse(
                    choices=[
                        CompletionChoiceResponse(
                            end=None if running_state[i] else int(datetime.now().timestamp() * 1000),
                            text=batch_output[i] if response_text_handler is None else response_text_handler(
                                batch_output[i]),
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
                    interrupted=is_interrupted
                )
                yield response
                is_interrupted = response.interrupted
            index += 1
            new_mask = torch.tensor(running_state, dtype=torch.int8).unsqueeze(1).to(device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            if position_ids is not None:
                position_ids = position_ids[:, -1:].clone() + 1

        del past_key_values, out, batch_input_ids, attention_mask, position_ids
        gc.collect()
        torch.cuda.empty_cache()

    @abstractmethod
    def embedding(self):
        pass
