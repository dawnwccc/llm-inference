from typing import Tuple
from torchinfo import summary
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from config import ServerConfig
import os
from abc import abstractmethod

from serve.models.base_model import AbstractModelFunction
from serve.utils.factory import GlobalFactory, register_model_adapter


def load_model(
        model_name_or_path: str, device, **from_pretrained_kwargs
    ) -> Tuple[PreTrainedTokenizer, PreTrainedModel, AbstractModelFunction]:
    debug = from_pretrained_kwargs.pop("debug", False)
    tokenizer, model = (GlobalFactory.
                        get_model_adapter(model_name_or_path).
                        load_model(model_name_or_path, device, **from_pretrained_kwargs))
    model_function = GlobalFactory.get_model_function(model_name_or_path)
    if debug:
        # summary(model)
        print(tokenizer)
        print(model)
    return tokenizer, model, model_function


class ModelAdapter:

    @classmethod
    @abstractmethod
    def load_model(cls, model_name_or_path: str, device, **from_pretrained_kwargs):
        pass


# @register_model_adapter("default")
class DefaultModelAdapter(ModelAdapter):
    """默认模型适配器"""

    @classmethod
    def load_model(cls, model_name_or_path: str, device, **from_pretrained_kwargs):
        revision = from_pretrained_kwargs.get("revision", "main")
        use_fast = from_pretrained_kwargs.get("use_fast", True)
        trust_remote_code = from_pretrained_kwargs.get("trust_remote_code", True)

        if os.path.exists(os.path.join(ServerConfig.MODEL_CACHED_PATH, model_name_or_path)):
            model_name_or_path = os.path.join(ServerConfig.MODEL_CACHED_PATH, model_name_or_path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                revision=revision,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                revision=revision,
                use_fast=False,
                trust_remote_code=trust_remote_code
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=True,
                **from_pretrained_kwargs
            )
        except NameError:
            model = AutoModel.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=True,
                **from_pretrained_kwargs
            )

        return tokenizer, model


# @register_model_adapter("chatglm")
class ChatGLModelAdapter(ModelAdapter):

    @classmethod
    def load_model(cls, model_name_or_path: str, device, **from_pretrained_kwargs):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, revision=revision
        )
        model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True, **from_pretrained_kwargs
        )
        return tokenizer, model


def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""
    import torch

    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024 ** 3)
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory
