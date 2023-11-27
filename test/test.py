import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from serve.utils.inference.batcher import batch_tokenize

checkpoint = r"C:\Research\llm_code_quality_research\models\chatglm3-6b"
device = "cuda"  # for GPU usage or "cpu" for CPU usage
texts = [
    "def print_hello(",
    "def bubble_sort("
]

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
input_ids = torch.as_tensor([30910, 33030, 34797], dtype=torch.int32)
print(input_ids)
output = tokenizer.batch_decode(input_ids)
print(output)
# model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
# request = {"n": 1,
#            "temperature": 0,
#            "max_tokens": 20,
#            "stop_str": ["if __name", "def", "# "]}
# batch_inputs = batch_tokenize(prompt=texts, n=1, device=device, tokenize_func=tokenizer)
# print(batch_inputs)
