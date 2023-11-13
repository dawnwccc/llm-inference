import logging
from datetime import datetime
from threading import Thread
from utils.logger import Logger
import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
import uvicorn
from protocol.api_protocol import ModelRegisterRequest, ModelHeartBeatRequest, BaseResponse
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

    def run(self, app: FastAPI, host: str = "127.0.0.1", port: int = 8000, log_level=logging.WARNING):
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

    # def _kill_session(self, model_url, model_func, session_id: Union[str, List[str]]):
    #     try:
    #         response = httpx.post(url=f"{model_url}/kill_session", json={
    #             "session_id": session_id,
    #             "model_func": model_func
    #         }, timeout=Config.timeout)
    #         response.raise_for_status()
    #         self.logger.info(f"session: {session_id} is killed successfully.")
    #         return [session_id] if isinstance(session_id, str) else session_id
    #     except Exception as e:
    #         self.logger.error(f"session: {session_id} is not killed. reason: {e}, retry: {session_id}")
    #         return []
    #
    # def kill_session(self, session_id: Union[str, List[str]]):
    #     session_id = session_id
    #     success_session_id_list = []
    #     if isinstance(session_id, str):
    #         if session_id not in self.session_list.keys():
    #             return LLMServerResponse().success().set_message(f"{session_id} has been killed.")
    #         model_name, model_func = self.session_list.get(session_id)
    #         if model_name not in self.model_list.keys():
    #             self.session_list.pop(session_id)
    #             return LLMServerResponse().success().set_message(f"{session_id} has been killed.")
    #         model_info = self.model_list.get(model_name)
    #         model_url = model_info.get("model_url")
    #         success_session_id_list = self._kill_session(model_url, model_func, session_id)
    #     else:
    #         # isinstance(session_id, Iterable)
    #         model2id = {}
    #         for sid, session_info in self.session_list.items():
    #             if sid in session_id:
    #                 model_session_list = model2id.get(session_info, [])
    #                 model_session_list.append(sid)
    #             model2id[session_info] = model_session_list
    #         for (model_name, model_func), sid_list in model2id.items():
    #             if model_name not in self.model_list.keys():
    #                 for sid in sid_list:
    #                     self.session_list.pop(sid)
    #             model_info = self.model_list.get(model_name)
    #             model_url = model_info.get("model_url")
    #             success_session_list = self._kill_session(model_url, model_func, session_id)
    #             success_session_id_list.extend(success_session_list)
    #     return LLMServerResponse().success().set_message(f"session: {success_session_id_list} be killed successfully.")
    #
    # def generate(self, model_name, model_kwargs, session_id):
    #     if model_name not in self.model_list.keys():
    #         return (LLMServerResponse().error()
    #                 .set_message(f"{model_name} is not available."))
    #     model_info = self.model_list.get(model_name)
    #     model_url = model_info.get("model_url")
    #     if "generate" not in model_info["model_inference"]:
    #         return LLMServerResponse().error().set_message(f"{model_name} is not support generate.")
    #     self.session_list[session_id] = (model_name, "generate")
    #     try:
    #         model_kwargs["session_id"] = session_id
    #         response = httpx.post(url=f"{model_url}/generate", json=model_kwargs, timeout=Config.timeout)
    #         response.raise_for_status()
    #         self.logger.info(
    #             f"session: {session_id} {model_name} generate {model_kwargs} -> {response.json()}")
    #         self.session_list.pop(session_id)
    #         return response.json()
    #     except Exception as e:
    #         self.logger.error(
    #             f"session: {session_id} {model_name} generate failure, model_kwargs: {model_kwargs}, reason: {e}")
    #         self.session_list.pop(session_id)
    #         return LLMServerResponse().error().set_message(
    #             f"session: {session_id} {model_name} generate failure, model_kwargs: {model_kwargs}, reason: {e}")
    #
    # def embedding(self, model_name, model_kwargs, session_id):
    #     if model_name not in self.model_list.keys():
    #         return (LLMServerResponse().error()
    #                 .set_message(f"{model_name} is not available."))
    #     model_info = self.model_list.get(model_name)
    #     model_url = model_info.get("model_url")
    #     if "embeddings" not in model_info["model_inference"]:
    #         return LLMServerResponse().error().set_message(f"{model_name} is not support embeddings.")
    #     self.session_list[session_id] = (model_name, "embedding")
    #     try:
    #         response = httpx.post(url=f"{model_url}/embeddings", json=model_kwargs, timeout=Config.timeout)
    #         response.raise_for_status()
    #         self.logger.info(f"{model_name} embeddings {model_kwargs} -> {response.json()}")
    #         self.session_list.pop(session_id)
    #         return response.json()
    #     except Exception as e:
    #         self.logger.error(
    #             f"session: {session_id} {model_name} embeddings failure, model_kwargs: {model_kwargs}, reason: {e}")
    #         self.session_list.pop(session_id)
    #         return LLMServerResponse().error().set_message(
    #             f"session: {session_id} {model_name} embeddings failure, model_kwargs: {model_kwargs}, reason: {e}")
    #
    # def run(self, app, host="127.0.0.1", port=8000, reload=False, workers=1):
    #     # 初始化并执行定时器
    #     self._scheduler_thread = Thread(target=self.__init_schedule)
    #     # 设为守护线程，主线程结束则该线程也会结束
    #     self._scheduler_thread.daemon = True
    #     self._scheduler_thread.start()
    #     # 启动FastAPI
    #     uvicorn.run(app=app, host=host, port=port, reload=reload, workers=workers)
    #
    # def __init_schedule(self):
    #     self._scheduler.add_job(self.__send_heart_beat, "interval", seconds=self.config.heart_beat_second)
    #     self._scheduler.start()
    #
    # def __send_heart_beat(self):
    #     for model_name, model_info in copy.deepcopy(self.model_list).items():
    #         try:
    #             response = httpx.get(url=model_info["model_url"] + self.config.heart_beat_url, timeout=Config.timeout)
    #             response.raise_for_status()
    #             if response.json().get("state", False):
    #                 self.model_list[model_name]["failure_count"] = 0
    #                 # self.logger.info(f"{model_name} send heart beat success")
    #             else:
    #                 if model_info["failure_count"] >= self.config.heart_beat_failure_max_count:
    #                     self.model_list.pop(model_name)
    #                     self.logger.error(
    #                         f"{model_name} send heart beat failure, will be removed, current models: {list(self.model_list.keys())}. reason: {e}")
    #                     continue
    #                 self.model_list[model_name]["failure_count"] += 1
    #                 self.logger.info(f"{model_name} send heart beat failure")
    #         except Exception as e:
    #             if model_info["failure_count"] >= self.config.heart_beat_failure_max_count:
    #                 self.model_list.pop(model_name)
    #                 self.logger.error(
    #                     f"{model_name} send heart beat failure, will be removed, current models: {list(self.model_list.keys())}. reason: {e}")
    #                 continue
    #             self.model_list[model_name]["failure_count"] += 1
    #             self.logger.error(f"{model_name} send heart beat failure. reason: {e}")


app = FastAPI()


@app.get("/model/list")
def get_model_list():
    model_list = server.get_model_list()
    return BaseResponse().success().set_data("models", model_list)


@app.get("/model/status")
def get_model_status():
    model_list = server.get_model_status()
    return BaseResponse().success().set_data("models", model_list)


#
#
# @app.post(Config.register_url)
# def register(request: LLMServiceCenterRegisterRequest):
#     server.logger.info(f"register {request.dict()}")
#     server.register_model(request.model_name, request.dict())
#     return LLMServiceCenterHeartBeatResponse(state=True)
#
#
# @app.post("/generate")
# def generate(request: LLMServiceCenterRequest):
#     return server.generate(request.model_name, request.model_kwargs, request.session_id)
#
#
# @app.post("/embeddings")
# def embeddings(request: LLMServiceCenterRequest):
#     return server.embedding(request.model_name, request.model_kwargs, request.session_id)
#
#
# @app.post("/kill_session")
# def kill_session(request: LLMServiceKillSessionRequest):
#     return server.kill_session(request.session_id)


if __name__ == "__main__":
    server = LLMServerCenter()
    server.run(app, host=ServerConfig.SERVER_CENTER_URL, port=ServerConfig.SERVER_CENTER_PORT)
