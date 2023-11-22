import logging
from datetime import datetime
from typing import Dict, Any, Union, List

from fastapi.exceptions import RequestValidationError

from serve.entity.exception import GlobalException, global_exception_handler, request_validation_exception_handler, \
    exception_handler
from serve.utils.logger import Logger
import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Request
import uvicorn
from serve.entity.protocol.api_protocol import ModelRegisterRequest, ModelHeartBeatRequest, BaseResponse
from config import ServerConfig


class LLMServerCenter:
    """
    大模型注册中心
    server_list:{
        "server_name": {
            "state": "online" / "offline"
            "server_url": "http://xxx",
            "server_function": ["text_completion", "chat.completion" ...]
        }
    }
    """

    def __init__(self):
        self.server_list = {}
        self.logger = Logger.build_logger("logs", "llm_server_center")
        # 定时器
        self.scheduler = BackgroundScheduler()
        self.session_list = {}
        self.max_heartbeat_interval = 3 * ServerConfig.MAX_HEARTBEAT_FAILURES * ServerConfig.HEARTBEAT_RATE * 1000

    def run(self, app: FastAPI, host: str = "127.0.0.1", port: int = 8000, log_level=logging.DEBUG):
        # 开始定时器
        self.scheduler.add_job(self.check_heartbeat, "interval", seconds=ServerConfig.HEARTBEAT_RATE)
        self.scheduler.start()
        # 添加注册和心跳监听
        app.add_api_route(path=ServerConfig.REGISTER_URL,
                          endpoint=self.receive_register_request,
                          methods=["POST"])
        app.add_api_route(path=ServerConfig.HEARTBEAT_URL,
                          endpoint=self.receive_heartbeat_request,
                          methods=["POST"])
        app.add_exception_handler(GlobalException, global_exception_handler)
        app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
        uvicorn.run(app=app, host=host, port=port, log_level=log_level)

    def receive_register_request(self, request: ModelRegisterRequest):
        if not request.id.startswith("reg"):
            return BaseResponse().error().set_message("无效的请求")
        self.server_list[request.server_name] = {
            "state": "online",
            "server_url": request.server_url,
            "server_function": request.server_function,
            "update_time": int(datetime.now().timestamp() * 1000)
        }
        return BaseResponse().success()

    def check_heartbeat(self):
        for server_name, server_info in self.server_list.items():
            if int(datetime.now().timestamp() * 1000) - server_info["update_time"] > self.max_heartbeat_interval:
                server_info["state"] = "offline"
                self.logger.warning(f"{server_name} is offline.")

    def receive_heartbeat_request(self, request: ModelHeartBeatRequest):
        if not request.id.startswith("hb"):
            return BaseResponse().error().set_message("无效的请求")
        if request.server_name not in self.server_list.keys():
            return BaseResponse().error().set_message("未注册的模型")
        if self.server_list[request.server_name]["state"] == "offline":
            return BaseResponse().error().set_message("模型过期")
        self.server_list[request.server_name]["update_time"] = int(datetime.now().timestamp() * 1000)
        return BaseResponse().success()

    def get_model_list(self):
        return [item[0] for item in filter(lambda item: item[1]["state"] == "online", self.server_list.items())]

    def get_model_status(self):
        model_list = []
        for k, v in self.server_list.items():
            server_info = {"server_name": k}
            server_info.update(v)
            model_list.append(server_info)
        return model_list

    def check_request(self, params: Dict[str, Any]):
        model = params.get("model", None)
        if model is None:
            raise GlobalException("model is required.")
        if model not in self.server_list:
            raise GlobalException(message=f"model {model} is not available")

    @exception_handler
    def completions(self, params: Dict[str, Any]):
        self.check_request(params)
        model = params.get("model")
        response = httpx.post(url=f"{self.server_list[model]['server_url']}/v1/completions",
                              json=params, timeout=ServerConfig.SESSION_TIMEOUT)
        return response.json()

    @exception_handler
    def chat_completions(self, params: Dict[str, Any]):
        self.check_request(params)
        model = params.get("model")
        response = httpx.post(url=f"{self.server_list[model]['server_url']}/v1/chat/completions",
                              json=params, timeout=ServerConfig.SESSION_TIMEOUT)
        return response.json()

    @exception_handler
    def send_kill_signal(self, model: str, session_id: Union[str, List[str]]):
        if model not in self.server_list:
            raise GlobalException(f"model {model} can't be killed.")
        response = httpx.post(url=f"{self.server_list[model]['server_url']}{ServerConfig.KILL_SIGNAL_URL}",
                              json={
                                  "model": model,
                                  "session_id": session_id
                              }, timeout=ServerConfig.SESSION_TIMEOUT)
        return response.json()


app = FastAPI()


@app.get("/model/list")
def get_model_list():
    model_list = server.get_model_list()
    return BaseResponse().success().set_data("models", model_list)


@app.get("/model/status")
def get_model_status():
    model_list = server.get_model_status()
    return BaseResponse().success().set_data("models", model_list)


@app.post("/v1/completions")
async def completions(request: Request):
    # TODO: analysis headers to avoid to spiders
    params: dict = await request.json()
    return server.completions(params)


@app.post("/v1/chat/completions")
async def completions(request: Request):
    # TODO: analysis headers to avoid to spiders
    params: dict = await request.json()
    return server.chat_completions(params)


@app.post(ServerConfig.KILL_SIGNAL_URL)
async def kill_completion(request: Request):
    params: dict = await request.json()
    model = params.get("model", None)
    session_id = params.get("session_id", None)
    if model is None:
        return BaseResponse().error().set_message("model is required.")
    if session_id is None:
        return BaseResponse().error().set_message("session_id is required.")
    return server.send_kill_signal(model, session_id)


if __name__ == "__main__":
    server = LLMServerCenter()
    server.run(app, host=ServerConfig.SERVER_CENTER_URL, port=ServerConfig.SERVER_CENTER_PORT)
