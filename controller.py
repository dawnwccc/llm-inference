from serve.entity.exception import GlobalException
from serve.entity.inference import CompletionParams
from serve.entity.protocol import CompletionRequest, BaseResponse, ChatCompletionRequest
from serve.model_server import BaseModelServer
from fastapi import FastAPI
from serve.utils.init import init_environment

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
    # server = BaseModelServer("pycoder258k", r"C:\Projects\Python\my-llm-utils\model\iter258k", "cpu",
    #                          debug=True)
    # server = BaseModelServer("pycoder258k", r"H:\Projects\Python\models\python258k", "cuda",
    #                          revision="main", debug=True)
    server = BaseModelServer("chatglm3", r"C:\Research\llm_code_quality_research\models\chatglm3-6b", "cpu",
                             revision="main", debug=True)
    # server = BaseModelServer("chatglm3", r"G:\chatglm3", "cpu",
    #                          revision="main", debug=True)
    server.run(app, port=8001)
