'''
Pytorch complementation
Output sentence tensors and calculate their similarity
'''

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

# Designate device
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# Set model as bert-base-chinese
model = BertModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "[CLS] 我们 想 感谢 一些 人 [SEP] [MASK] 这些 人 帮助 了 我们 [SEP]"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Create the tensors of segments
segments_ids = [0] * 10 + [1] * 10

# Convert tensors to Pytorch tensors
tokens_tensor = torch.tensor([indexed_tokens]).to(device)
segments_tensors = torch.tensor([segments_ids]).to(device)

# Set mode to evaluation
model.eval()
model.to(device)

# Get MASK index
def get_index1(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]
masked_index = get_index1(tokenized_text, '[MASK]')

# Get encoded_layers
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)

# Confirm the shape of model
print ("Number of layers:", len(encoded_layers))	
layer_i = 0	
print ("Number of batches:", len(encoded_layers[layer_i]))	
batch_i = 0	
print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))	
token_i = 0	
print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

# Extract the feature values of [MASK]
# Defaultly set character-level mask
token_i = masked_index[0]

# Convert the hidden state embeddings into single token vectors		
# Encoded_layers have the shape: [# tokens, # layers, # features]	
token_embeddings = []
for token_i in range(len(tokenized_text)):
    hidden_layers = []
    for layer_i in range(len(encoded_layers)):
        vec = encoded_layers[layer_i][batch_i][token_i]
        hidden_layers.append(vec)
    token_embeddings.append(hidden_layers)

# Get modified vectors
concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # vec_concatenated, [number_of_tokens, 3072]	
summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # vec_summed, [number_of_tokens, 768]


# Adding Connectives as comparision

# Add connective because
text_because = "[CLS] 我们 想 感谢 一些 人 [SEP] 因 这些 人 帮助 了 我们 [SEP]"
tokenized_text_because = tokenizer.tokenize(text_because)
indexed_tokens_because = tokenizer.convert_tokens_to_ids(tokenized_text_because)

# Create the tensors of segments
segments_ids_because = [0] * 10 + [1] * 10

# Convert tensors to Pytorch tensors
tokens_tensor_because = torch.tensor([indexed_tokens_because]).to(device)
segments_tensors_because = torch.tensor([segments_ids_because]).to(device)

# Set mode to evaluation
model.eval()
model.to(device)

# Get MASK index
def get_index1(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]
masked_index = get_index1(tokenized_text, '[MASK]')

# Predict tokens and set encoded_layers
with torch.no_grad():
    encoded_layers_because, _ = model(tokens_tensor_because, segments_tensors_because)

# Extract the feature values of [MASK]
# Defaultly set character-level mask
token_i = masked_index[0]

# Convert the hidden state embeddings into single token vectors		
# Encoded_layers have the shape: [# tokens, # layers, # features]	
token_embeddings_because = []
for token_i in range(len(tokenized_text)):
    hidden_layers_because = []
    for layer_i in range(len(encoded_layers_because)):
        vec = encoded_layers_because[layer_i][batch_i][token_i]
        hidden_layers_because.append(vec)
    token_embeddings_because.append(hidden_layers_because)

# Get modified vectors
concatenated_last_4_layers_because = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings_because] # vec_concatenated, [number_of_tokens, 3072]	
summed_last_4_layers_because = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings_because] # vec_summed, [number_of_tokens, 768]


# Add connective but
text_but = "[CLS] 我们 想 感谢 一些 人 [SEP] 但 这些 人 帮助 了 我们 [SEP]"
tokenized_text_but = tokenizer.tokenize(text_but)
indexed_tokens_but = tokenizer.convert_tokens_to_ids(tokenized_text_but)

# Create the tensors of segments
segments_ids_but = [0] * 10 + [1] * 10

# Convert tensors to Pytorch tensors
tokens_tensor_but = torch.tensor([indexed_tokens_but]).to(device)
segments_tensors_but = torch.tensor([segments_ids_but]).to(device)

# Set mode to evaluation
model.eval()
model.to(device)

# Get MASK index
def get_index1(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]
masked_index = get_index1(tokenized_text, '[MASK]')

# Predict tokens and set encoded_layers
with torch.no_grad():
    encoded_layers_but, _ = model(tokens_tensor_but, segments_tensors_but)

# Extract the feature values of [MASK]
# Defaultly set character-level mask
token_i = masked_index[0]

# Convert the hidden state embeddings into single token vectors		
# Encoded_layers have the shape: [# tokens, # layers, # features]	
token_embeddings_but = []
for token_i in range(len(tokenized_text)):
    hidden_layers_but = []
    for layer_i in range(len(encoded_layers_but)):
        vec = encoded_layers_but[layer_i][batch_i][token_i]
        hidden_layers_but.append(vec)
    token_embeddings_but.append(hidden_layers_but)

# Get modified vectors
concatenated_last_4_layers_but = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings_but] # vec_concatenated, [number_of_tokens, 3072]	
summed_last_4_layers_but = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings_but] # vec_summed, [number_of_tokens, 768]

# Compare [MASK] to because
print(cosine_similarity(summed_last_4_layers[10].reshape(1,-1), summed_last_4_layers_because[10].reshape(1,-1))[0][0])
# Compare [MASK] to but
print(cosine_similarity(summed_last_4_layers[10].reshape(1,-1), summed_last_4_layers_but[10].reshape(1,-1))[0][0])