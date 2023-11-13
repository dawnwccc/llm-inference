import os
import gc
from typing import List

import torch
from transformers import LogitsProcessorList, TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor, \
    TopPLogitsWarper, TopKLogitsWarper
from utils.enums import CompletionFinishReasonEnum
from utils.factory import register_stream_completion_function, register_embedding_function
from logits_processor import InvalidScoreLogitsProcessor
import warnings
from models.chatglm_model import chatglm_process_output


def default_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def check_stop_str(output: str, stop_str_list: List[str], check_start: int):
    for stop_str in stop_str_list:
        stop_str_pos = output.rfind(stop_str, check_start)
        if stop_str_pos > 0:
            output = output[:stop_str_pos]
            return True, output
    return False, output


@register_stream_completion_function("default")
@torch.inference_mode()
def default_stream_completion(model, tokenizer, prompt, params, device, context_length=2048, stream_interval=3):
    # 初始化参数
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))
    max_new_tokens = max(int(params.get("max_new_tokens", 256)), 0)
    stop_str_list = params.get("stop_str", None)
    # is or not print prompt
    echo = bool(params.get("echo", False))
    stop_token_ids = params.get("stop_token_ids", None) or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)
    # 是否输出logprobs
    logprobs = params.get("logprobs", None)
    is_logprobs = logprobs is not None
    current_token_text_offset = len_prompt
    logprobs_text_offsets = []
    logprobs_token_logprobs = []
    logprobs_tokens = []
    logprobs_top_logprobs = []

    # get logits processor
    logits_processor = default_logits_processor(temperature, repetition_penalty, top_p, top_k)

    input_ids = tokenizer(prompt).input_ids
    input_ids_length = len(input_ids)
    output_ids = list(input_ids) if echo else []

    if model.config.is_encoder_decoder:
        input_ids = input_ids[-context_length:]
    else:
        input_ids = input_ids[-(context_length - max_new_tokens - 1):]

    encoder_output = None
    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )
    else:
        start_ids = torch.as_tensor(
            [input_ids],
            device=device,
        )

    past_key_values = out = None
    current_token = None
    token_index = 0
    logits = None
    is_stopped = False
    is_interrupted = False
    finish_reason = None
    while not is_stopped and not is_interrupted:
        if token_index == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(start_ids, use_cache=True)
                logits = out.logits

        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor([[current_token]], device=device),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[current_token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values
                )
                logits = out.logits
        past_key_values = out.past_key_values

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
                token_text = tokenizer.decode(token)
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
            output = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            if stop_str_list:
                is_stopped, output = check_stop_str(output, stop_str_list, len_prompt if echo else 0)
                if is_stopped:
                    finish_reason = CompletionFinishReasonEnum.stop
            response = {
                "text": output,
                "logprobs": None if not is_logprobs else {
                    "text_offset": logprobs_text_offsets,
                    "token_logprobs": logprobs_token_logprobs,
                    "tokens": logprobs_tokens,
                    "top_logprobs": logprobs_top_logprobs
                },
                "usage": {
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": token_index,
                    "total_tokens": input_ids_length + token_index,
                },
                "finish_reason": finish_reason,
            }
            yield response
            is_interrupted = response.get("interrupted", False)

    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


@register_stream_completion_function("chatglm")
@torch.inference_mode()
def chatglm_stream_completion(model, tokenizer, prompt, params, device, context_length=2048, stream_interval=3):
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = max(int(params.get("max_tokens", 256)), 0)
    stop_str_list = params.get("stop_str", None)
    # is or not print prompt
    echo = bool(params.get("echo", False))
    stop_token_ids = params.get("stop_token_ids", None) or []
    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command("<|user|>"),
    ]
    stop_token_ids.extend(eos_token_id)

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    input_ids_length = len(inputs["input_ids"][0])
    if input_ids_length >= model.config.seq_length:
        warnings.warn(f"Input length larger than {model.config.seq_length}")
    gen_kwargs = {
        "max_length": max_new_tokens + input_ids_length + 1,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    token_index = 0
    is_stopped = False
    finish_reason = None
    output_ids = inputs["input_ids"][0] if echo else []
    for total_ids in model.stream_generate(**inputs, eos_token_id=stop_token_ids, **gen_kwargs):
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
            output = tokenizer.decode(output_ids)
            output = chatglm_process_output(output)
            if stop_str_list:
                is_stopped, output = check_stop_str(output, stop_str_list, len_prompt if echo else 0)
                if is_stopped:
                    finish_reason = CompletionFinishReasonEnum.stop

            response = {
                "text": output,
                "logprobs": None,
                "usage": {
                    "prompt_tokens": input_ids_length,
                    "completion_tokens": token_index,
                    "total_tokens": input_ids_length + token_index,
                },
                "finish_reason": finish_reason,
            }
            yield response
            is_interrupted = response.get("interrupted", False)
            if is_interrupted or is_stopped:
                break
    gc.collect()
    torch.cuda.empty_cache()


@register_stream_completion_function("openai")
@torch.inference_mode()
def openai_stream_completion(model, tokenizer, prompt, params, device, context_length=2048, stream_interval=3):
    pass