from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="None",
)
response1 = client.chat.completions.create(
    model="chatglm3-6b",
    messages=[
        {"role": "system", "content": "You are a helpful and harmless assistant"},
        {"role": "user", "content": "hello"}
    ],
    max_tokens=10
)
print(response1)
response2 = client.completions.create(
    model="chatglm3-6b",
    prompt="hello, how are you?",
    max_tokens=10
)
print(response2)
