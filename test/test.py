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
input_ids = torch.as_tensor([64790, 64792, 64794, 30910,    13,   809,   383, 22011, 10461, 30944,
         30966, 30932,   260,  1796,  3239,  2092,  7594,   422,  1192,   899,
         30923, 30930, 23833, 30930,  5741,   267,  2795, 30953, 30917,  8417,
          7724, 30930, 21911,  1227,  3478,  3536, 30930, 64795, 30910,    13,
         36474, 54591, 64796], dtype=torch.int32)
# print(input_ids)
output = tokenizer.decode(input_ids)
print(output)
# model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
# request = {"n": 1,
#            "temperature": 0,
#            "max_tokens": 20,
#            "stop_str": ["if __name", "def", "# "]}
# batch_inputs = batch_tokenize(prompt=texts, n=1, device=device, tokenize_func=tokenizer)
# print(batch_inputs)
