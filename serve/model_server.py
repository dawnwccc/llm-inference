import logging
import os.path
import sys
import shortuuid
from abc import abstractmethod
import httpx
from typing import List
from adapter import load_model
import torch
import uvicorn
from fastapi import FastAPI
from datetime import datetime
from utils import Logger
from config import ServerConfig
from apscheduler.schedulers.background import BackgroundScheduler
from protocol.api_protocol import ModelRegisterRequest, ModelHeartBeatRequest, CompletionRequest, \
    BaseResponse, CompletionChoiceResponse, CompletionLogprobs, CompletionUsageInfo, ChatCompletionResponse, \
    CompletionResponse, ChatCompletionChoiceResponse, ChatCompletionRequest, ChatMessage
from utils.factory import GlobalFactory
from utils.enums import ModelFunctionEnum
from models.base_model import AbstractModelFunction
# init
import inference
from utils import chat_template


class BaseModelServer:
    """
    基础LLM服务类
    """
    MODEL_FUNCTION: List[ModelFunctionEnum] = []

    def __init__(self, model_name: str, model_name_or_path: str, device: str, **kwargs):
        # 读取模型所需信息
        self.model_name = model_name
        self.model_name_or_path = model_name_or_path
        self.tokenizer = None
        self.model = None
        self.device = device
        self.torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        self.num_gpus = kwargs.pop("num_gpus", 1)
        self.load_8bit = kwargs.pop("load_8bit", False)
        self.cpu_offloading = kwargs.pop("cpu_offloading", False)
        self.device_map = kwargs.pop("device_map", "auto")
        self.kwargs = kwargs
        self.context_length = kwargs.pop("context_length", 2048)
        self.model_function: AbstractModelFunction = None
        # 注册模型所需信息
        self.register_flag = False
        self.client = None
        self.center_url = f"http://{ServerConfig.SERVER_CENTER_URL}:{ServerConfig.SERVER_CENTER_PORT}"
        self.model_url = None
        # 定时器
        self.heartbeat_scheduler = BackgroundScheduler()
        self.heartbeat_failure_count = 0
        # 日志
        self.logger = self.build_logger()
        # kill session
        self.sessions_to_kill = []

    def send_heartbeat(self):
        try:
            heartbeat_data = ModelHeartBeatRequest(
                id=f"hb-{shortuuid.uuid()}",
                ip=self.model_url.split(":")[0],
                server_name=self.model_name
            ).parse2dict()
            response = self.client.post(f"{ServerConfig.HEARTBEAT_URL}", json=heartbeat_data)
            response.raise_for_status()
            if response.json().get("state", False):
                # 心跳成功
                self.heartbeat_failure_count = 0
            else:
                # 数据返回成功，心跳失败
                self.register_flag = False
                self.logger.error(f"{self.model_name} heartbeat failure. message: {response.json()['message']}")
        except Exception as e:
            self.heartbeat_failure_count += 1
            self.logger.error(f"{self.model_name} heartbeat failure. reason: {e}")
            if self.heartbeat_failure_count > ServerConfig.MAX_HEARTBEAT_FAILURES:
                self.register_flag = False
                # time.sleep(3 * ServerConfig.HEARTBEAT_RATE)
                self.logger.error(f"{self.model_name} has been exceeded maximum "
                                  f"{ServerConfig.MAX_HEARTBEAT_FAILURES} retries")

    def register_model(self):
        try:
            register_data = ModelRegisterRequest(
                id=f"reg-{shortuuid.uuid()}",
                ip=self.model_url.split(":")[0],
                server_name=self.model_name,
                server_url=self.model_url,
                server_function=self.MODEL_FUNCTION
            ).parse2dict()
            response = self.client.post(f"http://{ServerConfig.REGISTER_URL}", json=register_data)
            response.raise_for_status()
            if response.json().get("state", False):
                # 启动心跳
                self.register_flag = True
                self.logger.info(f"{self.model_name} register success.")
                self.heartbeat_failure_count = 0
            else:
                self.logger.error(f"{self.model_name} register failure. message: {response.json()['message']}")
        except Exception as e:
            self.logger.error(f"{self.model_name} register failure. reason: {e}")
            # time.sleep(3 * ServerConfig.HEARTBEAT_RATE)

    def init_register_and_heartbeat(self):
        if self.register_flag:
            self.send_heartbeat()
        else:
            self.register_model()

    def run(self, app: FastAPI, host: str = "127.0.0.1", port: int = 8001, log_level=logging.WARNING):
        self.load_model()
        self.model_url = f"http://{host}:{port}"
        self.client = httpx.Client(base_url=self.center_url,
                                   headers={
                                       "Content-Type": "application/json"
                                   })
        self.heartbeat_scheduler.add_job(self.init_register_and_heartbeat, "interval",
                                         max_instances=1,
                                         seconds=ServerConfig.HEARTBEAT_RATE),
        self.heartbeat_scheduler.start()
        uvicorn.run(app=app, host=host, port=port, log_level=log_level)

    def load_model(self):
        # cpu_offloading 是指在加载大型模型时, 将部分计算从CPU转移到GPU或其他加速器上进行,
        # 以减轻CPU的计算压力, 提高系统的总体执行效率。
        if self.cpu_offloading:
            assert self.load_8bit, "The cpu-offloading feature can only be used while also using 8-bit-quantization"
            assert "linux" not in sys.platform, "CPU-offloading is only supported on linux-systems"
            assert not self.device.startswith("cuda"), "CPU-offloading is only enabled when using CUDA-devices"

        if self.device.startswith("cuda"):
            self.torch_dtype = torch.float16
            assert torch.cuda.is_available(), "GPU is not available"
            assert self.num_gpus > 0, "num_gpus must be greater than 0"
            if self.num_gpus > 1:
                assert torch.cuda.device_count() >= self.num_gpus, "Not enough GPUs available"
                if self.device_map is None:
                    self.device_map = "balanced_low_0"
        elif self.device.lower() == "cpu":
            self.torch_dtype = torch.float32
            self.logger.warning("CPU: using torch.float32 to load model")
        self.kwargs["offload_folder"] = os.path.join(self.model_name_or_path, "offload")
        self.tokenizer, self.model, model_function_class = load_model(model_name_or_path=self.model_name_or_path,
                                                                      device=self.device,
                                                                      torch_dtype=self.torch_dtype,
                                                                      load_in_8bit=self.load_8bit,
                                                                      device_map=self.device_map,
                                                                      **self.kwargs)
        self.model_function = model_function_class(self.tokenizer,
                                                   self.model,
                                                   self.device,
                                                   self.context_length)

    @abstractmethod
    def build_logger(self):
        return Logger(None, "base", is_control=True)

    def completion(self, params):
        session_id = params["id"]
        prompt = params["prompt"]
        n = params.get("n", 1)
        prompt_size = len(prompt) if isinstance(prompt, list) else 1
        response = CompletionResponse(
            id=params["id"],
            model=params["model"],
            choices=[
                CompletionChoiceResponse(
                    index=i,
                    start=int(datetime.now().timestamp() * 1000),
                    end=None,
                    text="",
                    logprobs=None,
                    finish_reason=None,
                    usage=None
                )
                for i in range(prompt_size * n)
            ],
            usage=CompletionUsageInfo()
        )
        for temp_response in self.model_function.stream_completion(prompt, params, stream_interval=3):
            if temp_response is None:
                break
            for index, temp_choice in enumerate(temp_response["choices"]):
                choice = response.choices[index]
                choice.text = temp_choice["text"]
                choice.logprobs = CompletionLogprobs(**temp_choice["logprobs"]) if temp_choice["logprobs"] else None
                choice.finish_reason = temp_choice["finish_reason"]
                if temp_choice["finish_reason"] and len(temp_choice["finish_reason"]) != 0:
                    response.choices[index].end = int(datetime.now().timestamp() * 1000)
                choice.usage = CompletionUsageInfo(**temp_choice["usage"])

            if session_id in self.sessions_to_kill:
                self.logger.info(f"kill session {session_id}, generate stop")
                self.sessions_to_kill.remove(session_id)
                temp_response["interrupted"] = True
            response.usage.prompt_tokens = sum(choice.usage.prompt_tokens for choice in response.choices)
            response.usage.completion_tokens = sum(choice.usage.completion_tokens for choice in response.choices)
            response.usage.total_tokens = sum(choice.usage.total_tokens for choice in response.choices)
        return response

    def chat_completion(self, params):
        session_id = params["id"]
        message_template = GlobalFactory.get_chat_template(params["model"])
        n = params.get("n", 1)
        params["stop_str"].extend(message_template.stop_str)
        response = ChatCompletionResponse(
            id=params["id"],
            model=params["model"],
            choices=[
                ChatCompletionChoiceResponse(
                    index=i,
                    start=int(datetime.now().timestamp() * 1000),  # 毫秒为单位
                    end=None,
                    message=ChatMessage(role=message_template.roles[-1], content=""),
                    logprobs=None,
                    finish_reason=None,
                    usage=None
                )
                for i in range(n)
            ],
            usage=CompletionUsageInfo()
        )
        for temp_response in self.model_function.stream_chat_completion(message_template,
                                                                        params["messages"],
                                                                        params,
                                                                        stream_interval=3):
            if temp_response is None:
                break
            for index, temp_choice in enumerate(temp_response["choices"]):
                choice = response.choices[index]
                choice.message.content = temp_choice["text"]
                choice.logprobs = CompletionLogprobs(**temp_choice["logprobs"]) if temp_choice["logprobs"] else None
                choice.finish_reason = temp_choice["finish_reason"]
                if temp_choice["finish_reason"] and len(temp_choice["finish_reason"]) != 0:
                    response.choices[index].end = int(datetime.now().timestamp() * 1000)
                choice.usage = CompletionUsageInfo(**temp_choice["usage"])
            if session_id in self.sessions_to_kill:
                self.logger.info(f"kill session {session_id}, generate stop")
                self.sessions_to_kill.remove(session_id)
                temp_response["interrupted"] = True
            response.usage.prompt_tokens = sum(choice.usage.prompt_tokens for choice in response.choices)
            response.usage.completion_tokens = sum(choice.usage.completion_tokens for choice in response.choices)
            response.usage.total_tokens = sum(choice.usage.total_tokens for choice in response.choices)
        return response


app = FastAPI()


@app.post("/v1/completions")
def completion(request: CompletionRequest):
    params = request.parse2dict()
    if len(params["prompt"]) == 0:
        return BaseResponse().error().message("messages can't be empty")
    out = server.completion(params)
    return BaseResponse().success().update_data(out)


@app.post("/v1/chat/completions")
def chat_completion(request: ChatCompletionRequest):
    params = request.parse2dict()
    if len(params["messages"]) == 0:
        return BaseResponse().error().set_message("messages can't be empty")
    out = server.chat_completion(params)
    return BaseResponse().success().update_data(out)


if __name__ == "__main__":
    # server = BaseModelServer("pycoder258k", r"C:\Projects\Python\my-llm-utils\model\iter258k", "cpu",
    #                          revision="main")
    server = BaseModelServer("chatglm", r"C:\Research\llm_code_quality_research\models\chatglm3-6b", "cpu",
                             revision="main", debug=True)
    server.run(app, port=8001)
