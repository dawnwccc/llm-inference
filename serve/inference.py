from typing import Union, List

import torch
from transformers import LogitsProcessorList, TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor, \
    TopPLogitsWarper, TopKLogitsWarper


def batch_pack(prompt: Union[str, List[str]], n: int = 1):
    """prompt * n"""
    n = max(n, 1)
    prompt_packed = []
    if isinstance(prompt, str):
        prompt_packed = list([prompt] for _ in range(n))
    else:
        for p in prompt:
            prompt_packed.append(list(p for _ in range(n)))
    return prompt_packed


def batch_tokenize(tokenizer, prompt: Union[str, List[str]], n: int, device: str, pad_token_id: int = 0):
    if tokenizer.pad_token_id:
        # 尽管 attention_mask 可以阻止，注意力更新，但使用不同的 pad_token_id 会导致不同的input_embeds
        pad_token_id = tokenizer.pad_token_id
    prompts = batch_pack(prompt, n)
    input_ids_list = []
    input_ids_length_list = []
    for p in prompts:
        input_ids = tokenizer(p)["input_ids"][0]
        input_ids_length_list.append(len(input_ids))
        input_ids_list.append(input_ids)
    max_length = max(input_ids_length_list)
    input_ids_packed = []
    attention_mask_packed = []
    for ids in input_ids_list:
        pad_len = max_length - len(ids)
        input_ids = ids + [pad_token_id] * pad_len
        attention_mask = [1] * len(ids) + [0] * pad_len
        input_ids_packed.append(input_ids)
        attention_mask_packed.append(attention_mask)
    return prompts, (torch.tensor(input_ids_packed).to(device), torch.tensor(attention_mask_packed).to(device))


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
