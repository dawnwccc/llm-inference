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


def default_stopping_criteria(
        max_length: int, max_time: float = None,
) -> CustomStoppingCriteriaList:
    criteria_list = CustomStoppingCriteriaList()
    if max_time:
        criteria_list.add(MaxTimeCriteria(max_time), CompletionFinishReasonEnum.time)
    criteria_list.add(MaxLengthCriteria(max_length), CompletionFinishReasonEnum.length)
    return criteria_list

