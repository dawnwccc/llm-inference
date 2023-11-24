import gc
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
from serve.utils.inference import default_logits_processor, check_stop_str, batch_tokenize
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

    def stream_completion(
            self, prompt: Union[str, List[str]], params: CompletionParams, stream_interval: int = 2
    ) -> TempCompletionResponse:
        """流式文本补全"""
        # TODO: 判断设备使用情况
        n = params.n
        if n == 1:
            if isinstance(prompt, str):
                return self.single_stream_completion(prompt, params, self.device, stream_interval)
            elif isinstance(prompt, list) and len(prompt) == 1:
                return self.single_stream_completion(prompt[0], params, self.device, stream_interval)
        return self.batch_stream_completion(prompt, params, self.device, stream_interval)

    @abstractmethod
    def single_stream_completion(
            self, prompt: Union[str, List[str]], params: CompletionParams, device: str, stream_interval: int = 2
    ) -> TempCompletionResponse:
        """流式文本补全"""
        pass

    @abstractmethod
    def batch_stream_completion(
            self, prompt: Union[str, List[str]], params: CompletionParams, device: str, stream_interval: int = 2
    ) -> TempCompletionResponse:
        """批次文本补全"""
        pass

    def stream_chat_completion(
            self, chat_template: ChatTemplate, messages: List[ChatMessage], params: CompletionParams, stream_interval: int = 2
    ) -> TempCompletionResponse:
        prompt = chat_template.complete_message(messages)
        return self.stream_completion(prompt, params, stream_interval)

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
    def single_stream_completion(
            self, prompt: Union[str, List[str]], params: CompletionParams, device: str, stream_interval: int = 2
    ) -> TempCompletionResponse:
        # 初始化参数
        len_prompt = len(prompt)
        temperature = params.temperature
        repetition_penalty = params.repetition_penalty
        top_p = params.top_p
        top_k = params.top_k
        max_new_tokens = params.max_tokens
        stop_str_list = params.stop_str
        # is or not print prompt
        echo = params.echo
        stop_token_ids = params.stop_token_ids
        if self.tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        # 是否输出logprobs
        logprobs = params.logprobs
        is_logprobs = logprobs is not None
        current_token_text_offset = len_prompt
        logprobs_text_offsets = []
        logprobs_token_logprobs = []
        logprobs_tokens = []
        logprobs_top_logprobs = []

        # get logits processor
        logits_processor = default_logits_processor(temperature, repetition_penalty, top_p, top_k)

        input_ids = self.tokenizer(prompt).input_ids
        input_ids_length = len(input_ids)
        output_ids = list(input_ids) if echo else []

        if self.model.config.is_encoder_decoder:
            input_ids = input_ids[-self.context_length:]
        else:
            input_ids = input_ids[-(self.context_length - max_new_tokens - 1):]

        encoder_output = None
        if self.model.config.is_encoder_decoder:
            encoder_output = self.model.encoder(
                input_ids=torch.as_tensor([input_ids], device=self.device)
            )[0]
            start_ids = torch.as_tensor(
                [[self.model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=self.device,
            )
        else:
            start_ids = torch.as_tensor(
                [input_ids],
                device=device,
            )

        past_key_values = None
        current_token = None
        token_index = 0
        is_stopped = False
        is_interrupted = False
        finish_reason = None
        while not is_stopped and not is_interrupted:
            if token_index == 0:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=start_ids,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(start_ids, use_cache=True)
                    logits = out.logits
            else:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=torch.as_tensor([[current_token]], device=device),
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(
                        input_ids=torch.as_tensor([[current_token]], device=device),
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    logits = out.logits
            past_key_values = out.past_key_values

            # 计算prompt的logprobs
            if echo and is_logprobs and token_index == 0:
                prompt_token_offsets = 0
                prompt_openai_logprobs = torch.log_softmax(logits[0], dim=-1)
                _, indices = torch.topk(logits[0], logprobs, dim=-1)
                for prompt_token_index in range(len(input_ids) - 1):
                    prompt_next_token_index = prompt_token_index + 1
                    prompt_next_token = input_ids[prompt_next_token_index]
                    tokens = [tok.tolist() for tok in indices[prompt_next_token_index]]
                    prompt_next_token_text = self.tokenizer.decode(prompt_next_token)
                    top_logprobs = {}
                    for token in tokens:
                        token_text = self.tokenizer.decode(token)
                        top_logprobs[token_text] = prompt_openai_logprobs[prompt_token_index, token].tolist()
                    prompt_token_offsets += len(prompt_next_token_text)
                    logprobs_tokens.append(prompt_next_token_text)
                    logprobs_token_logprobs.append(
                        prompt_openai_logprobs[prompt_token_index, prompt_next_token].tolist())
                    logprobs_text_offsets.append(prompt_token_offsets)
                    logprobs_top_logprobs.append(top_logprobs)

            if logits_processor:
                last_token_logits = logits_processor(
                    torch.as_tensor([output_ids], device=logits.device).long(),
                    logits[:, -1, :]
                )[0]
            else:
                last_token_logits = logits[0, -1, :]

            if is_logprobs:
                if temperature < 1e-5 or top_p < 1e-8:
                    probs = last_token_logits
                    _, indices = torch.topk(last_token_logits, logprobs)
                else:
                    probs = torch.softmax(last_token_logits, dim=-1)
                    indices = torch.multinomial(probs, num_samples=logprobs)
                tokens = [int(i) for i in indices.tolist()]
                openai_logprobs = torch.log_softmax(logits[0, -1, :], dim=-1).tolist()
                probs = probs.tolist()
                top_logprobs = {}
                max_token_probs = float("-inf")
                max_token_id = None
                max_token_text = None
                for token in tokens:
                    openai_logprobs_value = openai_logprobs[token]
                    token_text = self.tokenizer.decode(token)
                    top_logprobs[token_text] = openai_logprobs_value
                    if probs[token] > max_token_probs:
                        max_token_id = token
                        max_token_text = token_text
                        max_token_probs = probs[token]
                current_token_text_offset += len(max_token_text)
                logprobs_tokens.append(max_token_text)
                logprobs_token_logprobs.append(top_logprobs[max_token_text])
                logprobs_text_offsets.append(current_token_text_offset)
                logprobs_top_logprobs.append(top_logprobs)
                current_token = max_token_id
            else:
                if temperature < 1e-5 or top_p < 1e-8:
                    current_token = int(torch.argmax(last_token_logits))
                else:
                    probs = torch.softmax(last_token_logits, dim=-1)
                    current_token = int(torch.multinomial(probs, num_samples=1))
            if token_index < max_new_tokens:
                token_index += 1
                output_ids.append(current_token)
                if token_index == max_new_tokens:
                    is_stopped = True
                    finish_reason = CompletionFinishReasonEnum.length
            else:
                is_stopped = True
                finish_reason = CompletionFinishReasonEnum.length

            if current_token in stop_token_ids:
                is_stopped = True
                finish_reason = CompletionFinishReasonEnum.stop

            if token_index % stream_interval == 0 or is_stopped:
                output = self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                if stop_str_list:
                    stop_str_stopped, output = check_stop_str(output, stop_str_list, len_prompt if echo else 0)
                    if stop_str_stopped:
                        is_stopped = True
                        finish_reason = CompletionFinishReasonEnum.stop

                response = TempCompletionResponse(
                    choices=[CompletionChoiceResponse(
                        text=output,
                        logprobs=None if not is_logprobs else CompletionLogprobs(
                            text_offset=logprobs_text_offsets,
                            token_logprobs=logprobs_token_logprobs,
                            tokens=logprobs_tokens,
                            top_logprobs=logprobs_top_logprobs
                        ),
                        usage=CompletionUsageInfo(
                            prompt_tokens=input_ids_length,
                            completion_tokens=token_index,
                            total_tokens=input_ids_length + token_index
                        ),
                        finish_reason=finish_reason
                    )],
                    interrupted=False
                )
                yield response
                is_interrupted = response.interrupted

        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def batch_stream_completion(
            self, prompt: Union[str, List[str]], params: CompletionParams, device: str, stream_interval: int = 2
    ) -> TempCompletionResponse:
        # 初始化参数
        temperature = params.temperature
        repetition_penalty = params.repetition_penalty
        top_p = params.top_p
        top_k = params.top_k
        max_new_tokens = params.max_tokens
        stop_str_list = params.stop_str
        # is or not print prompt
        echo = params.echo
        stop_token_ids = params.stop_token_ids
        if self.tokenizer.eos_token_id not in stop_token_ids:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        n = params.n
        prompts, (input_ids, attention_mask) = batch_tokenize(self.tokenizer, prompt, n, device)
        task_total = len(input_ids)
        batch_prompt_length = [len(p[0]) for p in prompts]
        batch_input_ids_length = [sum(attn_mask.tolist()) for attn_mask in attention_mask]
        batch_output_ids = [list(input_ids[i]) if echo else [] for i in range(task_total)]
        # 是否输出logprobs
        logprobs = params.logprobs
        is_logprobs = logprobs is not None
        batch_logprobs = [{
            "current_token_text_offset": batch_prompt_length[i],
            "text_offset": [],
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": []
        } for i in range(task_total)]

        # construct logits processor
        logits_processor = default_logits_processor(temperature, repetition_penalty, top_p, top_k)

        if self.model.config.is_encoder_decoder:
            input_ids = input_ids[-self.context_length:]
        else:
            input_ids = input_ids[-(self.context_length - max_new_tokens - 1):]

        encoder_output = None
        if self.model.config.is_encoder_decoder:
            encoder_output = self.model.encoder(
                input_ids=torch.as_tensor(input_ids, device=device)
            )[0]
            start_ids = torch.as_tensor(
                [[self.model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=device,
            )
        else:
            start_ids = torch.as_tensor(
                input_ids,
                device=device,
            )

        current_token = None
        token_index = [0 for _ in range(task_total)]
        past_key_values = None
        running_state = [True for _ in range(task_total)]
        finish_reason = ["" for _ in range(task_total)]
        is_interrupted = False
        index = 0

        while any(running_state) and not is_interrupted:
            if index == 0:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=start_ids,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(start_ids,
                                     attention_mask=attention_mask,
                                     use_cache=True)
                    logits = out.logits

            else:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=torch.as_tensor(current_token, device=device),
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(
                        input_ids=torch.as_tensor(current_token, device=device),
                        attention_mask=attention_mask,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    logits = out.logits
            past_key_values = out.past_key_values

            # 计算prompt的logprobs
            if echo and is_logprobs and index == 0:
                prompt_token_offsets = [0 for _ in range(task_total)]
                batch_openai_logprobs = torch.log_softmax(logits, dim=-1)
                _, indices = torch.topk(batch_openai_logprobs, logprobs, dim=-1)
                for i in range(task_total):
                    if not running_state[i]:
                        continue
                    for prompt_token_index in range(len(input_ids[i]) - 1):
                        if attention_mask[i][prompt_token_index] == 0:
                            break
                        prompt_next_token_index = prompt_token_index + 1
                        prompt_next_token = input_ids[i, prompt_next_token_index]
                        prompt_next_token_text = self.tokenizer.decode(prompt_next_token)
                        tokens = [tok.tolist() for tok in indices[i, prompt_next_token_index]]
                        top_logprobs = {}
                        for token in tokens:
                            token_text = self.tokenizer.decode(token)
                            top_logprobs[token_text] = batch_openai_logprobs[i, prompt_token_index, token].tolist()
                        prompt_token_offsets[i] += len(prompt_next_token_text)
                        batch_logprobs[i]["tokens"].append(prompt_next_token_text)
                        batch_logprobs[i]["token_logprobs"].append(
                            batch_openai_logprobs[i, prompt_token_index, prompt_next_token].tolist())
                        batch_logprobs[i]["text_offset"].append(prompt_token_offsets[i])
                        batch_logprobs[i]["top_logprobs"].append(top_logprobs)

            last_token_logits = []
            if logits_processor:
                for i in range(task_total):
                    a = logits_processor(
                        torch.as_tensor([batch_output_ids[i]], device=logits.device).long(),
                        logits[i, -1, :]
                    )
                    last_token_logits.append(a)
                last_token_logits = torch.stack(last_token_logits).to(device)
            else:
                last_token_logits = logits[..., -1, :]

            current_token = []
            if is_logprobs:
                if temperature < 1e-5 or top_p < 1e-8:
                    probs = last_token_logits
                    _, indices = torch.topk(last_token_logits, logprobs, dim=-1)
                else:
                    probs = torch.softmax(last_token_logits, dim=-1)
                    indices = torch.multinomial(probs, num_samples=logprobs)
                for i in range(task_total):
                    if not running_state[i]:
                        continue
                    tokens = [int(tok) for tok in indices[i].tolist()]
                    openai_logprobs = torch.log_softmax(logits[i, -1, :], dim=-1).tolist()
                    probs_i = probs[i].tolist()
                    top_logprobs = {}
                    max_token_probs = float("-inf")
                    max_token_id = None
                    max_token_text = None
                    for token in tokens:
                        openai_logprobs_value = openai_logprobs[token]
                        token_text = self.tokenizer.decode(token)
                        top_logprobs[token_text] = openai_logprobs_value
                        if probs_i[token] > max_token_probs:
                            max_token_id = token
                            max_token_text = token_text
                            max_token_probs = probs_i[token]
                    batch_logprobs[i]["current_token_text_offset"] += len(max_token_text)
                    batch_logprobs[i]["tokens"].append(max_token_text)
                    batch_logprobs[i]["token_logprobs"].append(top_logprobs[max_token_text])
                    batch_logprobs[i]["text_offset"].append(batch_logprobs[i]["current_token_text_offset"])
                    batch_logprobs[i]["top_logprobs"].append(top_logprobs)
                    current_token.append(max_token_id)
            else:
                if temperature < 1e-5 or top_p < 1e-8:
                    current_token.extend(torch.argmax(last_token_logits, dim=-1).tolist())
                else:
                    probs = torch.softmax(last_token_logits, dim=-1)
                    current_token = [tok[0] for tok in torch.multinomial(probs, num_samples=1).tolist()]

            if index < max_new_tokens:
                index += 1
                for i in range(task_total):
                    if not running_state[i]:
                        continue
                    if current_token[i] in stop_token_ids:
                        running_state[i] = False
                        finish_reason[i] = CompletionFinishReasonEnum.stop.value
                    batch_output_ids[i].append(current_token[i])
                    token_index[i] += 1

            if index == max_new_tokens:
                running_state = [False if r else r for r in running_state]
                finish_reason = [CompletionFinishReasonEnum.length.value if len(r) == 0 else r for r in finish_reason]

            if index % stream_interval == 0 or not all(running_state):
                batch_output = [
                    self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
                    for output_ids in batch_output_ids
                ]

                if stop_str_list:
                    for i in range(task_total):
                        is_stopped, batch_output[i] = check_stop_str(batch_output[i],
                                                                     stop_str_list,
                                                                     batch_prompt_length[i] if echo else 0)
                        if is_stopped:
                            running_state[i] = False
                            finish_reason[i] = CompletionFinishReasonEnum.stop.value

                response = TempCompletionResponse(
                    choices=[
                        CompletionChoiceResponse(
                            text=batch_output[i],
                            logprobs=None if not is_logprobs else CompletionLogprobs(
                                text_offset=batch_logprobs[i]["text_offset"],
                                token_logprobs=batch_logprobs[i]["token_logprobs"],
                                tokens=batch_logprobs[i]["tokens"],
                                top_logprobs=batch_logprobs[i]["top_logprobs"]
                            ),
                            usage=CompletionUsageInfo(
                                prompt_tokens=batch_input_ids_length[i],
                                completion_tokens=token_index[i],
                                total_tokens=batch_input_ids_length[i] + token_index[i]
                            ),
                            finish_reason=finish_reason[i]
                        )
                        for i in range(task_total)
                    ],
                    interrupted=False
                )
                # response = {
                #     "choices": [
                #         {
                #             "text": batch_output[i],
                #             "logprobs": None if not is_logprobs else {
                #                 "text_offset": batch_logprobs[i]["text_offset"],
                #                 "token_logprobs": batch_logprobs[i]["token_logprobs"],
                #                 "tokens": batch_logprobs[i]["tokens"],
                #                 "top_logprobs": batch_logprobs[i]["top_logprobs"]
                #             },
                #             "usage": {
                #                 "prompt_tokens": batch_input_ids_length[i],
                #                 "completion_tokens": token_index[i],
                #                 "total_tokens": batch_input_ids_length[i] + token_index[i]
                #             },
                #             "finish_reason": finish_reason[i]
                #         }
                #         for i in range(task_total)
                #     ]
                # }
                yield response
                is_interrupted = response.interrupted
            current_token = torch.tensor(current_token, dtype=torch.int32).unsqueeze(1).to(device)
            new_mask = torch.tensor(running_state, dtype=torch.int8).unsqueeze(1).to(device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def embedding(self):
        pass
