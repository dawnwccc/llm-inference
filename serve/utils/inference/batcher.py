from typing import Union, List
import torch


def batch_pack(prompt: Union[str, List[str]], n: int = 1):
    """prompt * n"""
    n = max(n, 1)
    prompt_packed = []
    if isinstance(prompt, str):
        prompt_packed = list(prompt for _ in range(n))
    else:
        for p in prompt:
            prompt_packed.extend(list(p for _ in range(n)))
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
        input_ids = inputs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        input_ids_lengths.append(len(input_ids))
        batch_input_ids.append(input_ids)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()
            if isinstance(attention_mask[0], list):
                attention_mask = attention_mask[0]
            batch_attention_mask.append(attention_mask)
        else:
            batch_attention_mask.append(attention_mask)
        position_ids = inputs.get("position_ids", None)
        if position_ids is not None:
            if isinstance(position_ids, torch.Tensor):
                position_ids = position_ids.tolist()
            if isinstance(position_ids[0], list):
                position_ids = position_ids[0]
            batch_position_ids.append(position_ids)
        else:
            batch_position_ids.append(position_ids)

    max_length = max(input_ids_lengths)
    for i in range(len(batch_input_ids)):
        input_ids = batch_input_ids[i]
        padding_length = max_length - len(input_ids)
        batch_input_ids[i] = input_ids + [pad_token_id] * padding_length
        attention_mask = batch_attention_mask[i]
        if attention_mask is not None:
            batch_attention_mask[i] = attention_mask + [0] * padding_length
        else:
            batch_attention_mask[i] = [1] * len(input_ids) + [0] * padding_length
        position_ids = batch_position_ids[i]
        if position_ids is not None:
            last_position_id = position_ids[-1]
            batch_position_ids[i] = position_ids + list(range(last_position_id+1, last_position_id+1+padding_length))
    return prompts, (
        torch.tensor(batch_input_ids).to(device),
        torch.tensor(batch_attention_mask).to(device),
        torch.tensor(batch_position_ids).to(device) if all(batch_position_ids) else None
    )
