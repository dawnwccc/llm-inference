from typing import List, Optional, Tuple, Union

import torch
from pydantic import BaseModel
from transformers import StoppingCriteriaList, add_start_docstrings, StoppingCriteria, MaxLengthCriteria, \
    MaxTimeCriteria
from transformers.generation.stopping_criteria import STOPPING_CRITERIA_INPUTS_DOCSTRING, MaxNewTokensCriteria

from serve.utils.enums import CompletionFinishReasonEnum


def check_stop_str(output: str, stop_str_list: List[str], check_start: int):
    for stop_str in stop_str_list:
        stop_str_pos = output.rfind(stop_str, check_start)
        if stop_str_pos > 0:
            output = output[:stop_str_pos]
            return True, output
    return False, output


class StoppingCriteriaMessage(BaseModel):
    stop: bool
    message: Union[str, CompletionFinishReasonEnum] = None


class CustomStoppingCriteriaList:

    def __init__(self):
        self._list = []
        self._messages = []

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
            self, input_ids: Union[torch.LongTensor, torch.Tensor], scores: torch.FloatTensor, **kwargs
    ) -> StoppingCriteriaMessage:
        response = StoppingCriteriaMessage(
            stop=False,
            message=""
        )
        for criteria, message in zip(self._list, self._messages):
            if criteria(input_ids, scores, **kwargs):
                response.stop = True
                response.message = message
                return response
        return response

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self._list:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None

    def add(self, criteria: StoppingCriteria, message: Union[str, CompletionFinishReasonEnum]) -> None:
        self._list.append(criteria)
        self._messages.append(message)

    def append(self, obj):
        raise NotImplementedError("please use add method")


class StopIdsCriteria(StoppingCriteria):

    def __init__(self, stop_ids: List[torch.Tensor], start_index: int, stream_interval: int):
        # [[xxx], [xxx]]
        self.start_index = start_index
        self.stop_ids = stop_ids
        self.stop_ids_length = [len(tokens) for tokens in stop_ids]
        self.stream_interval = stream_interval
        self.stop_ids_min_length = min(self.stop_ids_length)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        # input_ids: [tokens]
        output_ids = input_ids[self.start_index:]
        output_ids_length = len(output_ids)
        if len(output_ids) < self.stop_ids_min_length:
            return False
        for tokens, tokens_length in zip(self.stop_ids, self.stop_ids_length):
            if output_ids_length < tokens_length:
                return False
            pos = self._rfind(output_ids, tokens)
            if pos > 0:
                attention_mask = kwargs["attention_mask"]
                attention_mask[:self.start_index+pos] = 1
                return True
        return False

    def _rfind(self, output_ids, stop_ids):
        stop_ids_length = len(stop_ids)
        start = len(output_ids)-stop_ids_length
        end = start - self.stream_interval - 1
        for i in range(start, end, -1):
            if torch.equal(output_ids[i:i+stop_ids_length], stop_ids):
                return i
        return -1


def default_stopping_criteria(
        max_length: int, stream_interval: int, max_time: float = None,
        stop_token_ids: List[torch.Tensor] = None, start_index: int = 0
) -> CustomStoppingCriteriaList:
    criteria_list = CustomStoppingCriteriaList()
    if stop_token_ids:
        criteria_list.add(StopIdsCriteria(stop_token_ids, start_index, stream_interval), CompletionFinishReasonEnum.stop)
    if max_time:
        criteria_list.add(MaxTimeCriteria(max_time), CompletionFinishReasonEnum.time)
    criteria_list.add(MaxLengthCriteria(max_length), CompletionFinishReasonEnum.length)
    return criteria_list

