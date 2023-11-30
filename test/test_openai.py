from openai import OpenAI

# base_url = "https://api.chatanywhere.com.cn/v1"
base_url = "http://172.18.101.28:5901/v1"
client = OpenAI(
    base_url=base_url,
    api_key="sess-ytGZpQmYjyGN5YlTattZU3oYlHsztmNbp4zoLrc7",
)
response1 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful and harmless assistant"},
        {"role": "user", "content": "hello"}
    ],
    max_tokens=0
)
print(response1)
response2 = client.completions.create(
    model="gpt-3.5-turbo",
    prompt="hello, how are you?",
    max_tokens=0,
    logprobs=1
)
print(response2)
