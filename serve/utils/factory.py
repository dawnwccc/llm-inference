import warnings


def register_model_adapter(key):
    def wrapper(model_adapter):
        GlobalFactory.register_model_adapter(key, model_adapter)
        return model_adapter

    return wrapper


def register_model_function(key):
    def wrapper(model_function):
        GlobalFactory.register_model_function(key, model_function)
        return model_function

    return wrapper


def register_chat_template(key):
    def wrapper(chat_template):
        GlobalFactory.register_chat_template(key, chat_template)
        return chat_template

    return wrapper


class GlobalFactory:
    _global = {}
    _model_adapters = {}
    _model_functions = {}
    _chat_templates = {}

    @classmethod
    def register(cls, key, value):
        assert key not in cls._global.keys(), f"Duplicate register variable: {key}"
        cls._global[key] = value

    @classmethod
    def register_model_adapter(cls, key, model_adapter):
        if key in cls._model_adapters:
            warnings.warn(f"Duplicate register model adapter: {key}")
        cls._model_adapters[key] = model_adapter

    @classmethod
    def register_model_function(cls, key, model_function):
        if key in cls._model_functions:
            warnings.warn(f"Duplicate register model function: {key}")
        cls._model_functions[key] = model_function

    @classmethod
    def register_chat_template(cls, key, chat_template):
        if key in cls._chat_templates:
            warnings.warn(f"Duplicate register chat template: {key}")
        cls._chat_templates[key] = chat_template

    @classmethod
    def get(cls, key):
        return cls._global[key]

    @classmethod
    def get_model_adapter(cls, model_name_or_path):
        for model_name, model_adapter in cls._model_adapters.items():
            if model_name in model_name_or_path:
                return model_adapter
        if "default" in cls._model_adapters:
            return cls._model_adapters["default"]
        raise RuntimeError(f"Can't find model adapter for model: {model_name_or_path}")

    @classmethod
    def get_model_function(cls, model_name_or_path):
        for model_name, model_function in cls._model_functions.items():
            if model_name in model_name_or_path:
                return model_function
        if "default" in cls._model_functions:
            return cls._model_functions["default"]
        raise RuntimeError(f"Can't find model function for model: {model_name_or_path}")

    @classmethod
    def get_chat_template(cls, model_name_or_path):
        for model_name, chat_template in cls._chat_templates.items():
            if model_name in model_name_or_path:
                return chat_template
        if "default" in cls._chat_templates:
            return cls._chat_templates["default"]
        # raise RuntimeError(f"Can't find chat template for model: {model_name_or_path}")
        return None
