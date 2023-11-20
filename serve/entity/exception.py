from typing import Union
from fastapi import HTTPException
from starlette.responses import Response
from serve.entity.protocol.api_protocol import BaseRequest, BaseResponse
from serve.utils.enums import HTTPStatusCode


class GlobalException(Exception):
    """全局异常"""

    def __init__(self, message: str, code: Union[HTTPStatusCode, int] = HTTPStatusCode.ERROR):
        self.message = message
        self.code = code


# def exception_handler(func):
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except GlobalException as exp:
#             return BaseResponse().error().set_message(exp.message, exp.code)
#
#     return wrapper


async def global_exception_handler(request: BaseRequest, exp: GlobalException):
    resp_str = BaseResponse().error().set_message(exp.message, exp.code).parse2json()
    return Response(content=resp_str)
