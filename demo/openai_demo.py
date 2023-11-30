from openai import OpenAI

base_url = "your server url"
client = OpenAI(
    base_url=base_url,
    api_key="EMPTY",
)
response1 = client.chat.completions.create(
    model="chatglm3-6b",
    messages=[
        {"role": "system", "content": "You are a helpful and harmless assistant"},
        {"role": "user", "content": "hello"}
    ],
    max_tokens=50
)
print(response1)
response2 = client.completions.create(
    model="chatglm3-6b",
    prompt="hello, how are you?",
    max_tokens=50,
    logprobs=1
)
print(response2)
