import gc
import re
import warnings
from typing import Union, List, Dict, Any
import torch

from serve.entity.exception import GlobalException
from serve.entity.inference import TempCompletionResponse, CompletionParams
from serve.entity.protocol import CompletionChoiceResponse, CompletionUsageInfo, ChatMessage
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
        stop_token_id_tensor = torch.tensor(stop_token_ids).to(device)
        logits_processor = default_logits_processor(temperature, repetition_penalty, top_p, top_k)
        logits_processor.append(ChatGLMInvalidScoreLogitsProcessor())
        input_ids, attention_mask, position_ids = self.tokenizer([prompt], return_tensors="pt").to(device)
        input_ids_length = len(input_ids[0])
        model_kwargs = {
            "max_length": max_new_tokens + input_ids_length + 1,
            "do_sample": True if temperature > 1e-5 else False,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "use_cache": True
        }
        if input_ids_length >= self.model.config.seq_length:
            raise GlobalException(f"Input length larger than {self.model.config.seq_length}")

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        print(input_ids)
        print(unfinished_sequences)
        scores = None
        past_key_values = None
        while True:
            # forward pass to get next token
            out = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True
            )

            past_key_values = out.past_key_values
            logits = out.logits

            if logits_processor:
                last_token_logits = logits_processor(input_ids, logits[:, -1, :])
            else:
                last_token_logits = logits[:, -1, :]

            if temperature < 1e-5 or top_p < 1e-8:
                current_token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                current_token = int(torch.multinomial(probs, num_samples=1))

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            if return_past_key_values:
                yield input_ids, outputs.past_key_values
            else:
                yield input_ids
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break


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
