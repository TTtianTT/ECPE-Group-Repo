'''
Pytorch complementation
'''

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch

# Designate device
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# Set model as gpt
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def score(tokens_tensor):
    loss=model(tokens_tensor, labels=tokens_tensor)[0]
    return np.exp(loss.cpu().detach().numpy())

texts = ["We want to thank some people because those people helped me", "We want to thank some people but those people helped me"]
for text in texts:
    tokens_tensor = tokenizer.encode( text, add_special_tokens=False, return_tensors="pt")           
    print(text, score(tokens_tensor))