import gc
import re
import warnings
from typing import Union, List, Dict, Any
import torch

from serve.entity.exception import GlobalException
from serve.entity.inference import TempCompletionResponse, CompletionParams
from serve.entity.protocol import CompletionChoiceResponse, CompletionUsageInfo, ChatMessage, CompletionLogprobs
from serve.utils.chat_template import ChatTemplate
from serve.utils.factory import register_model_function
from serve.utils.inference import check_stop_str, default_logits_processor
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

    # def single_stream_completion(self, prompt: Union[str, List[str]], params: CompletionParams, device: str,
    #                              stream_interval: int = 2):
    #     len_prompt = len(prompt)
    #     temperature = params.temperature
    #     repetition_penalty = params.repetition_penalty
    #     top_p = params.top_p
    #     max_new_tokens = params.max_tokens
    #     stop_str_list = params.stop_str
    #     # is or not print prompt
    #     echo = params.echo
    #     stop_token_ids = params.stop_token_ids
    #     eos_token_id = [
    #         self.tokenizer.eos_token_id,
    #         self.tokenizer.get_command("<|user|>"),
    #     ]
    #     stop_token_ids.extend(eos_token_id)
    #
    #     inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
    #     input_ids_length = len(inputs["input_ids"][0])
    #     if input_ids_length >= self.model.config.seq_length:
    #         warnings.warn(f"Input length larger than {self.model.config.seq_length}")
    #     gen_kwargs = {
    #         "max_length": max_new_tokens + input_ids_length + 1,
    #         "do_sample": True if temperature > 1e-5 else False,
    #         "top_p": top_p,
    #         "repetition_penalty": repetition_penalty,
    #         "logits_processor": [ChatGLMInvalidScoreLogitsProcessor()],
    #     }
    #     if temperature > 1e-5:
    #         gen_kwargs["temperature"] = temperature
    #
    #     token_index = 0
    #     is_stopped = False
    #     finish_reason = None
    #     output_ids = inputs["input_ids"][0] if echo else []
    #     print(inputs["input_ids"])
    #     for total_ids in self.model.stream_generate(**inputs, eos_token_id=stop_token_ids, **gen_kwargs):
    #         current_token = total_ids.tolist()[0][-1]
    #
    #         if token_index < max_new_tokens:
    #             token_index += 1
    #             output_ids.append(current_token)
    #             if token_index == max_new_tokens:
    #                 is_stopped = True
    #                 finish_reason = CompletionFinishReasonEnum.length
    #         else:
    #             is_stopped = True
    #             finish_reason = CompletionFinishReasonEnum.length
    #
    #         if token_index % stream_interval == 0 or is_stopped:
    #             output = self.tokenizer.decode(output_ids)
    #             output = chatglm_process_output(output)
    #             if stop_str_list:
    #                 is_stopped, output = check_stop_str(output, stop_str_list, len_prompt if echo else 0)
    #                 if is_stopped:
    #                     finish_reason = CompletionFinishReasonEnum.stop
    #
    #             response = TempCompletionResponse(
    #                 choices=[
    #                     CompletionChoiceResponse(
    #                         text=output,
    #                         logprobs=None,
    #                         usage=CompletionUsageInfo(
    #                             prompt_tokens=input_ids_length,
    #                             completion_tokens=token_index,
    #                             total_tokens=input_ids_length + token_index
    #                         ),
    #                         finish_reason=finish_reason
    #                     )
    #                 ],
    #                 interrupted=False
    #             )
    #             yield response
    #             is_interrupted = response.interrupted
    #             if is_interrupted or is_stopped:
    #                 break
    #     gc.collect()
    #     torch.cuda.empty_cache()

    def single_stream_completion(self, prompt: Union[str, List[str]], params: CompletionParams, device: str,
                                 stream_interval: int = 2):
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
        eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.get_command("<|user|>"),
        ]
        stop_token_ids.extend(eos_token_id)
        # 是否输出logprobs
        logprobs = params.logprobs
        is_logprobs = logprobs is not None
        current_token_text_offset = len_prompt
        logprobs_text_offsets = []
        logprobs_token_logprobs = []
        logprobs_tokens = []
        logprobs_top_logprobs = []

        logits_processor = default_logits_processor(temperature, repetition_penalty, top_p, top_k)
        logits_processor.append(ChatGLMInvalidScoreLogitsProcessor())
        inputs = self.tokenizer([prompt], return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        position_ids = inputs.position_ids
        input_ids_length = len(input_ids[0])
        if input_ids_length >= self.model.config.seq_length:
            raise GlobalException(f"Input length larger than {self.model.config.seq_length}")

        output_ids = []
        current_token = None
        token_index = 0
        past_key_values = None
        is_stopped = False
        is_interrupted = False
        finish_reason = None
        while not is_stopped and not is_interrupted:
            if token_index == 0:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    # return_dict=True,
                    # output_attentions=False,
                    # output_hidden_states=False,
                    return_last_logit=True
                )
            else:
                out = self.model(
                    input_ids=torch.as_tensor([[current_token]], device=device),
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_last_logit=True
                )
            past_key_values = out.past_key_values
            logits = out.logits

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
            # else:
            if temperature < 1e-5 or top_p < 1e-8:
                current_token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                current_token = int(torch.multinomial(probs, num_samples=1))
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            # position_ids = torch.cat(
            #     [position_ids, new_position_id], dim=-1
            # )
            position_ids = torch.as_tensor([[new_position_id]], device=device)

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
        gc.collect()
        torch.cuda.empty_cache()

    def batch_stream_completion(self, prompt: Union[str, List[str]], params: Dict[str, Any], device: str,
                                stream_interval: int = 2):
        pass

    def stream_chat_completion(
            self, chat_template: ChatTemplate, messages: List[ChatMessage], params: CompletionParams,
            stream_interval: int = 2
    ) -> TempCompletionResponse:
        conversation = chat_template.parse(messages)
        history = []
        history.extend(conversation.system_messages)
        history.extend(conversation.few_show_messages)
        history.extend(conversation.history_messages)
        prompt = conversation.prompt

        len_prompt = len(prompt)
        temperature = params.temperature
        repetition_penalty = params.repetition_penalty
        top_p = params.top_p
        max_new_tokens = params.max_tokens
        stop_str_list = params.stop_str
        # is or not print prompt
        echo = params.echo
        stop_token_ids = params.stop_token_ids
        eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.get_command("<|user|>"),
            self.tokenizer.get_command("<|observation|>")
        ]
        stop_token_ids.extend(eos_token_id)
        inputs = self.tokenizer.build_chat_input(prompt, history=history, role="user").to(self.device)
        input_ids_length = len(inputs["input_ids"][0])
        if input_ids_length >= self.model.config.seq_length:
            warnings.warn(f"Input length larger than {self.model.config.seq_length}")
        gen_kwargs = {
            "max_length": max_new_tokens + input_ids_length + 1,
            "do_sample": True if temperature > 1e-5 else False,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "logits_processor": [ChatGLMInvalidScoreLogitsProcessor()],
        }
        if temperature > 1e-5:
            gen_kwargs["temperature"] = temperature

        token_index = 0
        is_stopped = False
        finish_reason = None
        output_ids = inputs["input_ids"][0] if echo else []
        for total_ids in self.model.stream_generate(**inputs, eos_token_id=stop_token_ids, **gen_kwargs):
            current_token = total_ids.tolist()[0][-1]

            if token_index < max_new_tokens:
                token_index += 1
                output_ids.append(current_token)
                if token_index == max_new_tokens:
                    is_stopped = True
                    finish_reason = CompletionFinishReasonEnum.length
            else:
                is_stopped = True
                finish_reason = CompletionFinishReasonEnum.length

            if token_index % stream_interval == 0 or is_stopped:
                output = self.tokenizer.decode(output_ids)
                output = chatglm_process_output(output)
                if stop_str_list:
                    is_stopped, output = check_stop_str(output, stop_str_list, len_prompt if echo else 0)
                    if is_stopped:
                        finish_reason = CompletionFinishReasonEnum.stop

                response = TempCompletionResponse(
                    choices=[
                        CompletionChoiceResponse(
                            text=output,
                            logprobs=None,
                            usage=CompletionUsageInfo(
                                prompt_tokens=input_ids_length,
                                completion_tokens=token_index,
                                total_tokens=input_ids_length + token_index
                            ),
                            finish_reason=finish_reason
                        )
                    ],
                    interrupted=False
                )
                yield response
                is_interrupted = response.interrupted
                if is_interrupted or is_stopped:
                    break
        gc.collect()
        torch.cuda.empty_cache()

    def embedding(self):
        pass
