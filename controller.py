from serve.entity.exception import GlobalException
from serve.entity.protocol import CompletionRequest, BaseResponse, ChatCompletionRequest
from serve.model_server import BaseModelServer
from fastapi import FastAPI

app = FastAPI()


@app.post("/v1/completions")
def completion(request: CompletionRequest):
    params = request.parse2dict()
    if len(params["prompt"]) == 0:
        raise GlobalException("prompt can't be empty")
    out = server.completion(params)
    return BaseResponse().success().set_data(out)


@app.post("/v1/chat/completions")
def chat_completion(request: ChatCompletionRequest):
    params = request.parse2dict()
    if len(params["messages"]) == 0:
        raise GlobalException("messages can't be empty")
    out = server.chat_completion(params)
    return BaseResponse().success().set_data(out)


if __name__ == "__main__":
    # server = BaseModelServer("pycoder258k", r"C:\Projects\Python\my-llm-utils\model\iter258k", "cpu",
    #                          revision="main")
    server = BaseModelServer("chatglm", r"H:\Projects\Python\models\python258k", "cuda",
                             revision="main", debug=True)
    server.run(app, port=8001)
