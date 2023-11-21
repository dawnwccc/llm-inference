from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Union, overload
from datetime import datetime

from serve.entity.exception import GlobalException
from serve.entity.inference import Conversation
from serve.entity.protocol import ChatMessage
from serve.utils.factory import register_chat_template


@dataclass
class ChatTemplate:
    _raw_roles: Tuple[str] = ("user", "assistant")
    _raw_system_role: str = "system"
    _raw_function_role: str = "observation"

    roles: Tuple[str] = ("user", "assistant")
    system_role: str = ""
    function_role: str = "observation"

    few_shot: List[ChatMessage] = None
    few_shot_template: str = "{few_shot}"

    system_message: str = ""
    system_template: str = "{system_role}: {system_message}"

    message_template: str = "{role}: {content}"
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None
    sep: str = "\n"
    sep2: str = None

    @classmethod
    def complete_message(cls, messages: List[ChatMessage]) -> str:
        """Get the message for generation."""
        # TODO: 完善 function call
        system_message = cls.system_message
        few_shot_message = ""
        message_prompt = ""
        if cls.few_shot:
            for index, message in enumerate(cls.few_shot):
                few_shot_message += cls.message_template.format(
                    role=message.role,
                    content=message.content
                ) + cls.sep
        for index, message in enumerate(messages):
            if message.role == cls._raw_system_role:
                system_message += message.content
            elif message.role.startswith("few_shot"):
                few_shot_role = message.role.split(":")[-1]
                if few_shot_role == cls._raw_roles[0]:
                    role = cls.roles[0]
                elif few_shot_role == cls._raw_roles[-1]:
                    role = cls.roles[-1]
                else:
                    raise GlobalException("Invalid few shot role. for example: few_shot:user or few_shot:assistant")
                few_shot_message += cls.message_template.format(
                    role=role,
                    content=message.content
                ) + cls.sep
            else:

                if message.role == cls._raw_roles[0]:
                    role = cls.roles[0]
                elif message.role == cls._raw_roles[-1]:
                    role = cls.roles[-1]
                else:
                    raise GlobalException("Invalid role. for example: system, user, assistant")
                message_prompt += cls.message_template.format(
                    role=role,
                    content=message.content if message.content else ""
                ) + cls.sep
        message_prompt += cls.message_template.format(role=cls.roles[-1], content="")
        system_prompt = cls.system_template.format(system_role=cls.system_role, system_message=system_message)
        few_shot_prompt = ""
        if len(few_shot_message) > 0:
            few_shot_prompt = cls.few_shot_template.format(few_shot=few_shot_message)
        if cls.sep2:
            prompt = f"{cls.sep2}".join([system_prompt, few_shot_prompt, message_prompt])
        else:
            prompt = f"{system_prompt}{few_shot_prompt}{message_prompt}"
        return prompt

    @classmethod
    def parse(cls, messages: List[ChatMessage]) -> Conversation:
        """Parse the messages to dict."""
        few_shot_message = []
        system_message = []
        history = []
        if cls.system_message:
            system_message.append({
                "role": cls.system_role,
                "content": cls.system_message
            })
        if cls.few_shot:
            for index, message in enumerate(cls.few_shot):
                few_shot_message.append({
                    "role": message.role,
                    "content": message.content
                })
        for index, message in enumerate(messages[:-1]):
            if message.role == cls._raw_system_role:
                system_message.append({
                    "role": cls.system_role,
                    "content": message.content
                })
            elif message.role.startswith("few_shot"):
                few_shot_role = message.role.split(":")[-1]
                if few_shot_role == cls._raw_roles[0]:
                    role = cls.roles[0]
                elif few_shot_role == cls._raw_roles[-1]:
                    role = cls.roles[-1]
                else:
                    raise GlobalException("Invalid few shot role. for example: few_shot:user or few_shot:assistant")
                few_shot_message.append({
                    "role": role,
                    "content": message.content
                })
            else:
                if message.role == cls._raw_roles[0]:
                    role = cls.roles[0]
                elif message.role == cls._raw_roles[-1]:
                    role = cls.roles[-1]
                else:
                    raise GlobalException("Invalid role. for example: system, user, assistant")
                history.append({
                    "role": role,
                    "content": message.content if message.content else ""
                })
        prompt = ""
        if messages[-1].role == cls._raw_roles[0]:
            prompt = messages[-1].content
        return Conversation(
            prompt=prompt,
            system_messages=system_message,
            few_shot_messages=few_shot_message,
            history_messages=history
        )

@register_chat_template("single_colon")
@register_chat_template("default")
class ChatTemplateWithSingleColon(ChatTemplate):
    system_template: str = "{system_message}\n"
    few_shot: List[ChatMessage] = None
    few_shot_template: str = """For Example:\n{few_shot}\n"""
    message_template: str = "{role}: {content}"
    stop_str: Union[str, List[str]] = ["user:"]


@register_chat_template("chatglm")
class ChatGLMTemplate(ChatTemplate):
    roles: Tuple[str] = ("user", "assistant")
    function_role = "observation"
    system_role = "system"
    system_template: str = """{system_role}\n {system_message}"""
    few_shot: List[ChatMessage] = None
    few_shot_template: str = """例如：\n{few_shot}\n"""
    message_template: str = "{role}\n {content}"
    stop_str: Union[str, List[str]] = ["<|user|>", "<|observation|>", "</s>"]


if __name__ == "__main__":
    messages = [
        ChatMessage(**{"role": "system", "content": "You are a helpful, respectful and honest assistant."}),
        ChatMessage(**{"role": "user", "content": "Hello!"}),
        ChatMessage(**{"role": "assistant", "content": "Hi!"}),
        ChatMessage(**{"role": "user", "content": "How are you!"}),
        ChatMessage(**{"role": "few_shot:user", "content": "How are you!"}),
        ChatMessage(**{"role": "few_shot:assistant", "content": "I'm fine, thank you and you?"})
    ]
    prompt = ChatGLMTemplate.complete_message(messages)
    print(prompt)
