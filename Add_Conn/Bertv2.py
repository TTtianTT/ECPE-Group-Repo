'''
Pytorch complementation
'''

import torch
from transformers import BertTokenizer, BertForMaskedLM

# Designate device
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# Set model as bert-base-uncased and use BertForMaskedLM
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "[CLS] We want to thank some people [SEP] [MASK] those people helped me [SEP]"
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Create the tensors of segments
# ['[CLS]', 'we', 'want', 'to', 'thank', 'some', 'people', '[SEP]', '[MASK]', 'those', 'people', 'helped', 'me', '[SEP]']
segments_ids = [0] * 8 + [1] * 6

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

# Get prediction
with torch.no_grad():
    # [1，14，30522] # [#batch, #word, #vocab]
    # Outputs are the probabilities of words
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
predictions = outputs[0]  # [1，14，30522] # [#batch, #word, #vocab]

# Predict single word
predicted_index = torch.argmax(predictions[0, masked_index]).item()

# Transform index into word
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)

# Probability distribution
print(torch.exp(predictions[0, masked_index])/torch.sum(torch.exp(predictions[0, masked_index])))