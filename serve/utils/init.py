from serve.models.base_model import DefaultModelFunction
from serve.models.chatglm_model import ChatGLM3ModelFunction
from serve.utils.adapter import DefaultModelAdapter, ChatGLModelAdapter
from serve.utils.factory import GlobalFactory
from serve.utils.chat_template import *


def init_environment():
    # Model Adapter
    GlobalFactory.register_model_adapter("default", DefaultModelAdapter)
    GlobalFactory.register_model_adapter("chatglm", ChatGLModelAdapter)
    # Model Function
    GlobalFactory.register_model_function("default", DefaultModelFunction)
    GlobalFactory.register_model_function("chatglm3", ChatGLM3ModelFunction)
    # Chat Template