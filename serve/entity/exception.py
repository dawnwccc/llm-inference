from typing import Union

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.responses import Response
from serve.entity.protocol.api_protocol import BaseRequest, BaseResponse
from serve.utils.enums import HTTPStatusCode


class GlobalException(Exception):
    """全局异常"""

    def __init__(self, message: str, code: Union[HTTPStatusCode, int] = HTTPStatusCode.ERROR):
        self.message = message
        self.code = code


# def exception_handler(logger):
#     def wrapper1(func):
#         def wrapper2(*args, **kwargs):
#             try:
#                 return func(*args, **kwargs)
#             except ValidationError as exp:
#                 msg = [
#                     f"""{e["msg"]}: type: {e["type"]}, loc: {e["loc"][-1]}"""
#                     for e in exp.errors()
#                 ]
#                 raise GlobalException(message=";\n".join(msg), code=HTTPStatusCode.ERROR)
#             except BaseException as exp:
#                 raise GlobalException(message="internal system error", code=HTTPStatusCode.ERROR)
#         return wrapper2
#     return wrapper1


def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as exp:
            msg = [
                f"""{e["msg"]}: type: {e["type"]}, loc: {e["loc"][-1]}"""
                for e in exp.errors()
            ]
            raise GlobalException(message=";\n".join(msg), code=HTTPStatusCode.ERROR)
        except BaseException as exp:
            raise GlobalException(message="internal system error", code=HTTPStatusCode.ERROR)

    return wrapper


async def global_exception_handler(request: BaseRequest, exp: GlobalException):
    resp_str = BaseResponse().error().set_message(exp.message, exp.code).parse2json()
    return Response(content=resp_str)


async def request_validation_exception_handler(request: BaseRequest, exp: RequestValidationError):
    msg = [
        f"""{e["msg"]}: type: {e["type"]}, loc: {e["loc"][-1]}. Please check: {e["url"]}"""
        for e in exp.errors()
    ]
    resp_str = BaseResponse().error().set_message(";\n".join(msg), HTTPStatusCode.ERROR).parse2json()
    return Response(content=resp_str)