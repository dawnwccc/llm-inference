import openai
import os

# openai.api_key = "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"
openai.api_key = ""
openai.api_base = "http://127.0.0.1:8001/v1"
response1 = openai.ChatCompletion.create(
    model="chatglm3-6b",
    messages=[
        {"role": "system", "content": "You are a helpful and harmless assistant"},
        {"role": "user", "content": "hello"}
    ],
    max_tokens=10
)
print(response1)
response2 = openai.Completion.create(
    model="chatglm3-6b",
    prompt="hello, how are you?",
    max_tokens=10
)
print(response2)
