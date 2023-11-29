import json
from typing import Union, List, Dict, Optional, Any
from datetime import datetime

import shortuuid
from pydantic import BaseModel, Field
from serve.utils.enums import ModelFunctionEnum, HTTPStatusCode, CompletionFinishReasonEnum


class BaseRequest(BaseModel):
    """
    自定义大模型请求数据
    重写schema以及schema_json方法
    """
    id: Optional[str] = None
    ip: Optional[str] = None
    api_key: Optional[str] = ""
    version: Optional[Union[str, float, int]] = 1

    @classmethod
    def properties_schema(cls):
        properties = {}
        _properties = super().model_json_schema().get("properties")
        _required = super().model_json_schema().get("required")
        for property_key, property_info in _properties.items():
            if property_key in ("id", "ip", "api_key", "descr"):
                continue
            _property = {}
            if property_key in _required:
                _property["required"] = True
            else:
                _property["required"] = False
                _property["default"] = property_info.get("default")
            property_type = property_info.get("type")
            if property_type is None:
                property_type = []
                property_type_list = property_info.get("anyOf")
                for property_type_item in property_type_list:
                    if property_type_item["type"] == "array":
                        property_type.append(f"""array[{property_type_item["items"]["type"]}]""")
                    else:
                        property_type.append(property_type_item["type"])
                _property["type"] = property_type
            else:
                _property["type"] = [property_type]
            _property["descr"] = cls.descr.get(property_key, None)
            properties[property_key] = _property
        return properties

    @classmethod
    def properties_schema_json(cls):
        return json.dumps(cls.properties_schema())

    def parse2dict(self):
        return json.loads(self.model_dump_json())


class BaseResponse(BaseModel):
    """
    自定义大模型返回数据
    """
    message: str = None
    state: bool = None
    code: Union[int, HTTPStatusCode] = None
    data: Optional[Dict[str, Any]] = None

    def success(self):
        self.state = True
        self.code = HTTPStatusCode.OK
        if self.message is None:
            self.message = "success"
        return self

    def error(self):
        self.state = False
        self.code = HTTPStatusCode.ERROR
        if self.message is None:
            self.message = "unknown error"
        return self

    def set_data(self, value: Any = None, key: Union[str, dict, BaseModel] = None):
        if not self.data:
            self.data = {}
        if key and isinstance(key, str):
            self.data[key] = value
        else:
            if isinstance(value, dict):
                self.data.update(value)
            elif isinstance(value, BaseModel):
                self.data.update(value.model_dump())
            else:
                raise ValueError("value must be dict or BaseModel")
        return self

    def set_message(self, message: str, code: Union[int, HTTPStatusCode] = None):
        self.message = message
        if code:
            self.code = code
        return self

    def set_code(self, code: Union[int, HTTPStatusCode]):
        self.code = code
        return self

    def parse2dict(self):
        return json.loads(self.model_dump_json())

    def parse2json(self):
        return self.model_dump_json()


class ModelRegisterRequest(BaseRequest):
    server_name: str
    server_url: str
    server_function: List[str]


class ModelHeartBeatRequest(BaseRequest):
    server_name: str


class CompletionRequest(BaseRequest):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 16
    top_p: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.0
    stop_str: Optional[List[str]] = []
    echo: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = []
    max_context_len: Optional[int] = 2048
    stream_interval: Optional[int] = 3
    logprobs: Optional[int] = None
    n: Optional[int] = 1
    functions: Optional[Dict[str, str]] = None
    user: Optional[str] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


class EmbeddingsRequest(BaseRequest):
    """
    自定义大模型嵌入请求体
    """
    document: Union[str, List[str]]


class KillSignalRequest(BaseRequest):
    model: str
    session_id: Union[str, List[str]]
    # model_function: ModelFunctionEnum


class ModelPermission(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: f"modelperm-{shortuuid.random()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: Optional[str] = None
    object: str = "model"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    owned_by: str = ""
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []


class CompletionUsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class CompletionLogprobs(BaseModel):
    token_logprobs: List[float] = []
    tokens: List[str] = []
    top_logprobs: List[Dict[str, float]] = []
    text_offset: List[int] = []


class CompletionChoiceResponse(BaseModel):
    index: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = ""
    finish_reason: Optional[CompletionFinishReasonEnum] = None
    logprobs: Optional[CompletionLogprobs] = None
    usage: Optional[CompletionUsageInfo] = None


class CompletionResponse(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: ModelFunctionEnum = ModelFunctionEnum.completion
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[CompletionChoiceResponse] = []
    usage: Optional[CompletionUsageInfo] = None


class ChatMessage(BaseModel):
    role: str
    content: str
    # name: Optional[str] = None
    # function_call: Optional[object] = None


class ChatCompletionRequest(BaseRequest):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 16
    top_p: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.0
    stop_str: Optional[Union[str, List[str]]] = []
    stop_token_ids: Optional[List[int]] = []
    stream_interval: Optional[int] = 3
    logprobs: Optional[int] = None
    n: Optional[int] = 1
    functions: Optional[Dict[str, str]] = None
    user: Optional[str] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


class ChatCompletionChoiceResponse(BaseModel):
    index: int
    start: int
    end: Optional[int] = None
    message: Optional[ChatMessage] = []
    logprobs: Optional[CompletionLogprobs] = None
    finish_reason: Optional[CompletionFinishReasonEnum] = None
    usage: Optional[CompletionUsageInfo] = None


class ChatCompletionResponse(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    model: str
    object: ModelFunctionEnum = ModelFunctionEnum.chat_completion
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    choices: List[ChatCompletionChoiceResponse]
    usage: Optional[CompletionUsageInfo] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoiceResponse(BaseModel):
    index: int
    start: int
    end: Optional[int] = None
    delta: Optional[DeltaMessage] = None
    finish_reason: Optional[CompletionFinishReasonEnum] = None
    logprobs: Optional[CompletionLogprobs] = None
    usage: Optional[CompletionUsageInfo] = None
