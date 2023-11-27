import torch
from transformers import BertTokenizer, BertModel

# 初始化tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 假设我们的输入是两个句子
sentences = ['Hello world!', 'I love programming.']

# 对句子进行编码
encoded_inputs = tokenizer(sentences, padding='longest', return_tensors='pt')
print(encoded_inputs.input_ids)
print(tokenizer.batch_decode(encoded_inputs.input_ids.tolist()))

# 获取填充的token id
pad_token_id = tokenizer.pad_token_id
print(pad_token_id)

# 如果模型没有pad_token_id，我们可以手动设置一个
if model.config.pad_token_id is None:
    model.config.pad_token_id = pad_token_id

# 对输入进行填充
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']

# 调用模型
outputs = model(input_ids, attention_mask=attention_mask)
# print(outputs)
