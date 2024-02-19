from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')

for i in unmasker("AI will [MASK] the world."):
	print(i)
