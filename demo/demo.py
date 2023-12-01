import requests

# example http://127.0.0.1:8001/v1
base_url = "your server url"
completion_endpoint = base_url + "/completions"
chat_completion_endpoint = base_url + "/chat/completions"
headers = {
    "Content-Type": "application/json"
}
completion_request_data = {
    "model": "chatglm3-6b",
    "prompt": "def print_hello_world()",
    "temperature": 0.7,
    "repetition_penalty": 1.1,
    "max_tokens": 50,
    "logprobs": 1,
    "n": 1,
    "echo": True,
    "stop_str": ["//"]
}
# completion
response1 = requests.post(completion_endpoint, headers=headers, json=completion_request_data, timeout=300)
response1 = response1.json()
print(response1)

chat_completion_request_data = {
    "model": "chatglm3-6b",
    "messages": [
        {"role": "system", "content": "你是一个有用且无害的助手"},
        {"role": "user", "content": "你好"}
    ],
    "temperature": 0.7,
    "repetition_penalty": 1.1,
    "max_tokens": 50,
    "logprobs": 1,
    "n": 1,
    "echo": True,
    "stop_str": ["#"]
}
# chat completion
response2 = requests.post(chat_completion_endpoint, headers=headers, json=chat_completion_request_data, timeout=300)
response2 = response2.json()
print(response2)