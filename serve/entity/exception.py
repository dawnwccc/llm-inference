from typing import Union
import traceback
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import Response
from serve.entity.protocol.api_protocol import BaseRequest, BaseResponse
from serve.utils.enums import HTTPStatusCode


class GlobalException(Exception):
    """全局异常"""

    def __init__(self, message: str, code: Union[HTTPStatusCode, int] = HTTPStatusCode.ERROR, extra: str = None):
        self.message = message
        self.code = code
        self.extra = extra


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
        except GlobalException as exp:
            raise GlobalException(message=exp.message, code=exp.code, extra=exp.extra)
        except ValidationError as exp:
            msg = [
                f"""{e["msg"]}: type: {e["type"]}, loc: {e["loc"][-1]}"""
                for e in exp.errors()
            ]
            raise GlobalException(message=";\n".join(msg), code=HTTPStatusCode.ERROR, extra=str(exp))
        except BaseException as exp:
            # print(exp)
            trace = traceback.format_exc()
            print(trace)
            raise GlobalException(message="internal system error", code=HTTPStatusCode.ERROR, extra=trace)

    return wrapper


def global_exception_handler(logger):
    async def global_exception_handler_wrapper(request: Request, exp: GlobalException):
        log = f"{exp.message}."
        if exp.extra:
            log += f" detail: {exp.extra}"
        logger.error(log)
        resp_str = BaseResponse().error().set_message(exp.message, exp.code).parse2json()
        return Response(content=resp_str)

    return global_exception_handler_wrapper

# async def global_exception_handler(request: Request, exp: GlobalException):
#     # request_params = await request.json()
#     # logger.error(f"message: {exp.message}, request: {request_params}.")
#     resp_str = BaseResponse().error().set_message(exp.message, exp.code).parse2json()
#     return Response(content=resp_str)


def request_validation_exception_handler(logger):
    async def request_validation_exception_handler_wrapper(request: Request, exp: RequestValidationError):
        message = ";\n".join([
            f"""{e["msg"]}: type: {e["type"]}, loc: {e["loc"][-1]}. Please check: {e["url"]}"""
            for e in exp.errors()
        ])
        logger.error(f"{message}. detail: {str(exp)}")
        resp_str = BaseResponse().error().set_message(message, HTTPStatusCode.ERROR).parse2json()
        return Response(content=resp_str)

    return request_validation_exception_handler_wrapper
