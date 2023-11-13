def register_model_adapter(key):
    def wrapper(model_adapter):
        GlobalFactory.register_model_adapter(key, model_adapter)
        return model_adapter

    return wrapper


def register_stream_completion_function(key):
    def wrapper(stream_completion_function):
        GlobalFactory.register_stream_completion_function(key, stream_completion_function)
        return stream_completion_function

    return wrapper


def register_embedding_function(key):
    def wrapper(embedding_function):
        GlobalFactory.register_embedding_function(key, embedding_function)
        return embedding_function

    return wrapper


def register_chat_template(key):
    def wrapper(chat_template):
        GlobalFactory.register_chat_template(key, chat_template)
        return chat_template

    return wrapper


class GlobalFactory:
    _global = {}
    _model_adapters = {}
    _stream_completion_functions = {}
    _embedding_functions = {}
    _chat_templates = {}

    @classmethod
    def register(cls, key, value):
        assert key not in cls._global.keys(), f"Duplicate register variable: {key}"
        cls._global[key] = value

    @classmethod
    def register_model_adapter(cls, key, model_adapter):
        assert key not in cls._model_adapters.keys(), f"Duplicate register model adapter: {key}"
        cls._model_adapters[key] = model_adapter

    @classmethod
    def register_stream_completion_function(cls, key, stream_completion_function):
        assert key not in cls._stream_completion_functions.keys(), f"Duplicate register stream completion function: {key}"
        cls._stream_completion_functions[key] = stream_completion_function

    @classmethod
    def register_embedding_function(cls, key, embedding_function):
        assert key not in cls._embedding_functions.keys(), f"Duplicate register embedding function: {key}"
        cls._embedding_functions[key] = embedding_function

    @classmethod
    def register_chat_template(cls, key, chat_template):
        assert key not in cls._chat_templates.keys(), f"Duplicate register chat template: {key}"
        cls._chat_templates[key] = chat_template

    @classmethod
    def get(cls, key):
        return cls._global[key]

    @classmethod
    def get_model_adapter(cls, model_name_or_path):
        for model_name, model_adapter in cls._model_adapters.items():
            if model_name in model_name_or_path:
                return model_adapter
        if "default" in cls._model_adapters.keys():
            return cls._model_adapters["default"]
        raise RuntimeError(f"Can't find model adapter for model: {model_name_or_path}")

    @classmethod
    def get_stream_completion_function(cls, model_name_or_path):
        for model_name, stream_completion_function in cls._stream_completion_functions.items():
            if model_name in model_name_or_path:
                return stream_completion_function
        if "default" in cls._stream_completion_functions.keys():
            return cls._stream_completion_functions["default"]
        # raise RuntimeError(f"Can't find completion function for model: {model_name_or_path}")
        return None

    @classmethod
    def get_embedding_function(cls, model_name_or_path):
        for model_name, embedding_function in cls._embedding_functions.items():
            if model_name in model_name_or_path:
                return embedding_function
        if "default" in cls._embedding_functions.keys():
            return cls._embedding_functions["default"]
        # raise RuntimeError(f"Can't find embedding function for model: {model_name_or_path}")
        return None

    @classmethod
    def get_chat_template(cls, model_name_or_path):
        for model_name, chat_template in cls._chat_templates.items():
            if model_name in model_name_or_path:
                return chat_template
        if "default" in cls._chat_templates.keys():
            return cls._chat_templates["default"]
        # raise RuntimeError(f"Can't find chat template for model: {model_name_or_path}")
        return None
