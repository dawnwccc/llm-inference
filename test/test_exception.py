from fastapi import FastAPI, HTTPException
import uvicorn
from serve.entity.exception import GlobalException, global_exception_handler

app = FastAPI()


def test_exp_func():
    raise GlobalException(message="Item not found")
    # raise GlobalException("test")


@app.get("/exp")
def test_exp():
    test_exp_func()
    return {"msg": "ok"}


uvicorn.run(app, host="127.0.0.1", port=8000)
