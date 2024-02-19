import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
lines = open('fra.txt', 'r', encoding='utf-8').read().split('\n')

inputs, outputs = [], []

for k, line in enumerate(lines):
	inp, out, _ = line.split('\t')
	inputs.append(inp.lower())
	outputs.append(out.lower())
	
	if k > 1024 * 8:
		break
		
df = pd.DataFrame({'input' : inputs, 'target' : outputs})

class TranslationDatset(Dataset):
	def __init__(self, df, tokenizer, input_maxlen, target_maxlen):
		self.tokenizer     = tokenizer
		self.df            = df
		self.input_maxlen  = input_maxlen
		self.target_maxlen = target_maxlen
		
	def __len__(self):
		return len(self.df)
		
	def __getitem__(self, index):
		inputs  = self.df.iloc[index]['input']
		targets = self.df.loc[index]['target']
		
		inputs_tokens = self.tokenizer.encode_plus(inputs, 
								max_length=self.input_maxlen,
								truncation=True,
								padding='max_length',
								return_tensors='pt')
								
		
		targets_tokens = self.tokenizer.encode_plus(targets, 
								max_length=self.target_maxlen,
								truncation=True,
								padding='max_length',
								return_tensors='pt')

		return {'input_ids'              : inputs_tokens['input_ids'].squeeze(),
				'attention_mask'         : inputs_tokens['attention_mask'].squeeze(),
				'decoder_input_ids'      : targets_tokens['input_ids'].squeeze()[:-1],
				'decoder_attention_mask' : targets_tokens['attention_mask'].squeeze()[:-1],
				'labels'                 : targets_tokens['input_ids'].squeeze()[1:]}




# Model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model     = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

dataset = TranslationDatset(df, tokenizer, 128, 128)
dataset = DataLoader(dataset, batch_size=4)

# Hyperparameters
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

def save_model(model, epoch):
	model_path = os.path.join(models_dir, f'model_epoch_{epoch}.pt')
	torch.save(model.state_dict(), model_path)

def eval_model():
	model.eval()
	for example in df.sample(4).itertuples():
		inputs      = example.input
		input_ids   = tokenizer.encode(inputs, return_tensors='pt').to(device)
		targets_ids = model.generate(input_ids=input_ids, max_length=128, num_beams=3, repetition_penalty=2.5, length_penalty=1.0)
		output      = tokenizer.decode(targets_ids[0], skip_special_tokens=True)
		
		print(f"Text: '{inputs}' -> Translation: '{output}'  Target: '{example.target}'")

for epoch in range(1024):
	epoch_loss = 0.0
	with tqdm(dataset, desc=f'Epoch {epoch + 1}', unit='batch', leave=False) as tepoch:
		for i, batch in enumerate(tepoch):
			input_ids              = batch['input_ids'].to(device)
			attention_mask         = batch['attention_mask'].to(device)
			decoder_input_ids      = batch['decoder_input_ids'].to(device)
			decoder_attention_mask = batch['decoder_attention_mask'].to(device)
			labels                 = batch['labels'].to(device)
			
			outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
						decoder_input_ids=decoder_input_ids,decoder_attention_mask=decoder_attention_mask,
						labels=labels)

			loss = outputs[0]
			optimizer.zero_grad()
			loss.backward() # backpropagation
			optimizer.step()
			
			epoch_loss += loss.item()
			tepoch.set_postfix(losss=epoch_loss/(i + 1))
			
		save_model(model, epoch + 1)
		eval_model()
			






















