import openai
import os

# openai.api_key = "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"
openai.api_key = ""
openai.api_base = "http://127.0.0.1:8001/v1"
response = openai.ChatCompletion.create(
    model="chatglm3-6b",
    messages=[
        {"role": "system", "content": "You are a helpful and harmless assistant"},
        {"role": "user", "content": "hello"}
    ],
    max_tokens=10
)
# response = openai.Completion.create(
#     model="chatglm3-6b",
#     prompt="你好，请问你是谁？",
#     max_tokens=1
# )
print(response)
