from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

model     = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

# text input
text    = 'English to French: How much is it?'
inputs  = tokenizer(text, padding='longest', return_tensors='pt')

outputs = model.generate(**inputs)
output  = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(output)
