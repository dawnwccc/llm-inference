from serve.entity.exception import GlobalException
from serve.entity.inference import CompletionParams
from serve.entity.protocol import CompletionRequest, BaseResponse, ChatCompletionRequest
from serve.model_server import BaseModelServer
from fastapi import FastAPI
from serve.utils.init import init_environment

from argparse import ArgumentParser

# init
init_environment()

app = FastAPI()


@app.post("/v1/completions")
def completion(request: CompletionRequest):
    prompt = request.prompt
    params = CompletionParams(**request.parse2dict())
    if not prompt:
        raise GlobalException("prompt can't be empty")
    out = server.completion(prompt, params)
    return BaseResponse().success().set_data(out)


@app.post("/v1/chat/completions")
def chat_completion(request: ChatCompletionRequest):
    messages = request.messages
    params = CompletionParams(**request.parse2dict())
    if not messages:
        raise GlobalException("messages can't be empty")
    out = server.chat_completion(messages, params)
    return BaseResponse().success().set_data(out)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--revision", type=str, default="main")

    args = vars(parser.parse_args())
    # model_name = args.pop("model")
    # model_path = args.pop("model_path")
    device = args.pop("device")
    is_debug = args.pop("debug")
    device_map = args.pop("device_map")

    host = args.pop("host")
    port = args.pop("port")

    model_name = "chatglm3-6b"
    model_path = r"C:\Research\llm_code_quality_research\models\chatglm3-6b"
    device = "cpu"

    server = BaseModelServer(model_name=model_name, model_name_or_path=model_path, device=device,
                             debug=is_debug, device_map=device_map, **args)
    server.run(app, host=host, port=port)
