from typing import Union, List, Optional, Dict
from pydantic import BaseModel
from serve.entity.protocol import BaseRequest
from serve.utils.enums import OpenAICompletionObjectEnum, CompletionFinishReasonEnum


class OpenAICompletionRequest(BaseRequest):
    """request object of openai completion"""
    model: str
    prompt: Union[str, List[str]]
    best_of: Optional[int] = 1
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[dict] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    suffix: Optional[str] = None
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1
    user: Optional[str] = None


class OpenAICompletionLogprobs(BaseModel):
    text_offset: List[int]
    token_logprobs: List[float]
    tokens: List[str]
    top_logprobs: List[Dict[str, float]]


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAICompletionChoices(BaseModel):
    finish_reason: CompletionFinishReasonEnum
    index: int
    logprobs: Optional[OpenAICompletionLogprobs] = None
    text: str = ""
    start: int  # 额外
    end: int  # 额外
    usage: OpenAIUsage  # 额外


class OpenAICompletionResponse(BaseModel):
    id: str
    choices: List[OpenAICompletionChoices]
    created: int
    model: str
    object: OpenAICompletionObjectEnum
    usage: OpenAIUsage


class OpenAIChatCompletionRequest(BaseRequest):
    """request object of openai chat completion"""
    model: str
    messages: Union[str, List[Dict[str, str]]]
    frequency_penalty: Optional[float] = 0
    function_call: Optional[Union[str, Dict[str, str]]] = None
    functions: Optional[Dict[str, str]] = None  # 暂时并没有使用
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = 512
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1
    user: Optional[str] = None


class OpenAIChatCompletionMessage(BaseModel):
    content: Optional[str]
    role: Optional[str]
    function_call: Optional[Dict[str, str]] = None


class OpenAIChatCompletionChoices(BaseModel):
    finish_reason: CompletionFinishReasonEnum
    index: int
    message: OpenAIChatCompletionMessage
    logprobs: Optional[OpenAICompletionLogprobs] = None
    start: int  # 额外
    end: int  # 额外
    usage: OpenAIUsage  # 额外


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    choices: List[OpenAICompletionChoices]
    created: int
    model: str
    object: OpenAICompletionObjectEnum
    usage: OpenAIUsage
