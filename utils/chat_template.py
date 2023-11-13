from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Union
from datetime import datetime

from utils.factory import register_chat_template


@dataclass
class ChatTemplate:
    roles: Tuple[str] = ("user", "assistant")
    few_shot: List[Dict[str, str]] = None
    few_shot_template: str = "{few_shot}"
    system_message: str = ""
    system_template: str = "{system_message}"
    message_template: str = "{role}: {content}"
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None
    sep: str = "\n"
    sep2: str = None

    @classmethod
    @abstractmethod
    def complete_message(cls, messages: List[Dict[str, str]]) -> str:
        """Get the message for generation."""
        pass


@register_chat_template("single_colon")
@register_chat_template("default")
class ChatTemplateWithSingleColon(ChatTemplate):
    system_message: str = f"""Current date: {datetime.now().strftime("%Y-%m-%d")}\n"""
    few_shot: List[Dict[str, str]] = None
    few_shot_template: str = """For Example:\n{few_shot}\n"""
    message_template: str = "{role}: {content}"
    stop_str: Union[str, List[str]] = ["user:"]

    @classmethod
    def complete_message(cls, messages: List[Dict[str, str]]) -> str:
        """Get the message for generation."""
        system_message = cls.system_message
        few_shot_message = ""
        message_prompt = ""
        if cls.few_shot:
            for index, message in enumerate(cls.few_shot):
                few_shot_message += cls.message_template.format(
                    role=message["role"],
                    content=message["content"]
                ) + cls.sep
        for index, message in enumerate(messages):
            if message["role"] == "system":
                system_message += message["content"]
            elif message["role"].startswith("few_shot"):
                role = message["role"].split(":")[-1]
                few_shot_message += cls.message_template.format(
                    role=role,
                    content=message["content"]
                ) + cls.sep
            else:
                message_prompt += cls.message_template.format(
                    role=message["role"],
                    content=message["content"] if message["content"] else ""
                ) + cls.sep
        message_prompt += cls.message_template.format(role="assistant", content="")
        system_prompt = cls.system_template.format(system_message=cls.system_message)
        few_shot_prompt = ""
        if len(few_shot_message) > 0:
            few_shot_prompt = cls.few_shot_template.format(few_shot=few_shot_message)
        if cls.sep2:
            prompt = f"{cls.sep2}".join([system_prompt, few_shot_prompt, message_prompt])
        else:
            prompt = f"{system_prompt}{few_shot_prompt}{message_prompt}"
        return prompt


@register_chat_template("chatglm")
class ChatGLMTemplate(ChatTemplate):

    @classmethod
    def complete_message(cls, messages: List[Dict[str, str]]) -> str:
        pass


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you!"},
        {"role": "few_shot:user", "content": "How are you!"},
        {"role": "few_shot:assistant", "content": "I'm fine, thank you and you?"}
    ]
    prompt = ChatTemplateWithSingleColon.complete_message(messages)
    print(prompt)
