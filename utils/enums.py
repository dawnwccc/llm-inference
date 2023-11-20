from enum import Enum


class ModelFunctionEnum(Enum):
    completion = "text_completion"
    chat_completion = "chat.completion"
    embedding = "embedding"


class OpenAICompletionObjectEnum(Enum):
    completion = "text_completion"
    chat_completion = "chat.completion"
    chat_completion_chunk = "chat.completion.chunk"
    embedding = "embedding"


class CompletionFinishReasonEnum(Enum):
    stop = "stop"
    length = "length"
    interrupt = "interrupt"


class HTTPStatusCode(Enum):
    """
    HTTP Status Codes
    """
    OK = 5200  # 成功
    ERROR = 5500  # 服务端发生错误
