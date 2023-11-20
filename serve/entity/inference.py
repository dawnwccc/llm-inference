from typing import List, Optional, Union
from serve.entity.protocol import CompletionChoiceResponse
from pydantic import BaseModel, Field


class TempCompletionResponse(BaseModel):
    choices: List[CompletionChoiceResponse]
    interrupted: bool = False


class CompletionParams(BaseModel):
    id: str
    model: str
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = -1
    max_tokens: int = Field(default=16, ge=0, le=2048)
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    stop_str: Optional[Union[str, List[str]]] = []
    stop_token_ids: Union[List[str], List[int]] = []
    echo: bool = False
    logprobs: Optional[int] = None
    n: int = 1
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    # TODO: 完善 functions
    functions: object = None