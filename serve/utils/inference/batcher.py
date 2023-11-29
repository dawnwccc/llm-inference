from typing import Union, List

import torch


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


def batch_tokenize(
        prompt: Union[str, List[str]], n: int, device: str,
        tokenize_func, tokenize_func_kwargs=None,
        pad_token_id: int = 0,
):
    if tokenize_func_kwargs is None:
        tokenize_func_kwargs = {}
    prompts = batch_pack(prompt, n)

    input_ids_lengths = []
    batch_input_ids = []
    batch_attention_mask = []
    batch_position_ids = []
    for p in prompts:
        inputs = tokenize_func(p, **tokenize_func_kwargs)
        input_ids = inputs.get("input_ids")[0]
        input_ids_lengths.append(len(input_ids))
        batch_input_ids.append(input_ids)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask:
            batch_attention_mask.append(attention_mask[0])
        else:
            batch_attention_mask.append(attention_mask)
        position_ids = inputs.get("position_ids", None)
        if position_ids:
            batch_position_ids.append(position_ids[0])
        else:
            batch_position_ids.append(position_ids)

    max_length = max(input_ids_lengths)
    for i in range(len(batch_input_ids)):
        input_ids = batch_input_ids[i]
        padding_length = max_length - len(input_ids)
        batch_input_ids[i] = input_ids + [pad_token_id] * padding_length
        attention_mask = batch_attention_mask[i]
        if attention_mask:
            batch_attention_mask[i] = attention_mask + [0] * padding_length
        else:
            batch_attention_mask[i] = [1] * len(input_ids) + [0] * padding_length
        position_ids = batch_position_ids[i]
        if position_ids:
            last_position_id = position_ids[-1]
            batch_position_ids[i] = position_ids + list(range(last_position_id+1, last_position_id+1+padding_length))
    return prompts, (
        torch.tensor(batch_input_ids).to(device),
        torch.tensor(batch_attention_mask).to(device),
        torch.tensor(batch_position_ids).to(device) if all(batch_position_ids) else None
    )