# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer  = T5Tokenizer.from_pretrained("t5-small")
model      = T5ForConditionalGeneration.from_pretrained("t5-small", device_map="cuda")

input_text = 

input_ids  = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs    = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

