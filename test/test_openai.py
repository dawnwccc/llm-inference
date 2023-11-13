import openai

openai.api_key = "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"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful and harmless assistant"},
        {"role": "user", "content": "hello"}
    ],
    max_tokens=1
)
print(response)