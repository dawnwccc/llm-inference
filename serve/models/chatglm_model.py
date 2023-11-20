import gc
import re
import warnings
from typing import Union, List, Dict, Any
import torch

from serve.entity.inference import TempCompletionResponse, CompletionParams
from serve.entity.protocol import CompletionChoiceResponse, CompletionUsageInfo
from serve.utils.inference import check_stop_str
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


class ChatGLMModelFunction(AbstractModelFunction):

    def __init__(self, tokenizer, model, device, context_length):
        super().__init__(tokenizer, model, device, context_length)

    def single_stream_completion(self, prompt: Union[str, List[str]], params: CompletionParams, device: str,
                                 stream_interval: int = 2):
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
        ]
        stop_token_ids.extend(eos_token_id)

        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
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
                # response = {
                #     "text": output,
                #     "logprobs": None,
                #     "usage": {
                #         "prompt_tokens": input_ids_length,
                #         "completion_tokens": token_index,
                #         "total_tokens": input_ids_length + token_index,
                #     },
                #     "finish_reason": finish_reason,
                # }
                yield response
                is_interrupted = response.interrupted
                if is_interrupted or is_stopped:
                    break
        gc.collect()
        torch.cuda.empty_cache()

    def batch_stream_completion(self, prompt: Union[str, List[str]], params: Dict[str, Any], device: str,
                                stream_interval: int = 2):
        pass

    def chat_stream_completion(self, chat_template, prompt: Union[str, List[str]], params: Dict[str, Any], device: str,
                               stream_interval: int = 2):
        pass

    def embedding(self):
        pass
