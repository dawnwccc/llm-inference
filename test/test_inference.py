from serve.inference import default_stream_batch_completion
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = r"C:\Research\llm_code_quality_research\models\pycoder258k"
device = "cpu"  # for GPU usage or "cpu" for CPU usage
texts = [
    "def print_hello(",
    "def bubble_sort("
]

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
request = {"n": 1,
           "temperature": 0,
           "max_tokens": 20,
           "stop_str": ["if __name", "def", "# "]}
for i in default_stream_batch_completion(tokenizer, model, prompt=texts, params=request, device=device):
    pass
for k, v in i.items():
    print(v)
