import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'bert-base-cased'

model     = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# encode sentence to fixed vector
def get_embedding(sentence):
	tokens = tokenizer.encode(sentence, add_special_tokens=True)
	inputs = torch.tensor([tokens]) # 768
	with torch.no_grad():
		output = model(inputs) # compute the output
		last   = output.last_hidden_state
		embed  = torch.mean(last, dim=1)
	return embed


sentences = [
		'Sentença simples para teste',
		'BERT é bacana',
		'o gato pulou pela janela',
		'o carro é amarelo']
		
		
embeds = [get_embedding(x) for x in sentences]
		

while True:
	inputs = str(input('Digite uma frase: '))
	embed  = get_embedding(inputs)
	
	distances = [cosine_similarity(embed.numpy(), emb.numpy())[0][0] for emb in embeds]
	
	index = distances.index(max(distances)) # get the index of the closes sentence
	sent  = sentences[index] # get the sentence
	
	
	print('A senteça mais próxima de "{}" é "{}"'.format(inputs, sent))
	
	















