import csv
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM

# Init param
cause_uniconn = []
candidate_conn = []
with open ('../data/cause_uniconn_modified.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            cause_uniconn.append(word)
        line = f.readline()
with open ('../data/uniconn_modified.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            candidate_conn.append(word)
        line = f.readline()

# Load dataset
df = pd.read_csv('../data/pairs.csv')

# Init csv of result
with open ('../data/pairs_withconn.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['section', 'emo_clause_index', 'cau_candidate_index', 'emotion_clause', 'cause_candidate', 'conn', 'correctness', 'is_cause_conn'])

# Designate device
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# Set model as bert-base-uncased and use BertForMaskedLM
model = BertForMaskedLM.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

for i in range(len(df)):
    section = df['section'][i]
    emo_clause_index = df['emo_clause_index'][i]
    cau_candidate_index = df['cau_candidate_index'][i]
    emotion_clause = df['emotion_clause'][i]
    cause_candidate = df['cause_candidate'][i]
    text = '[CLS]' + str(emotion_clause) + '[SEP]' + '[MASK]' + str(cause_candidate) + '[SEP]'
    correctness = df['correctness'][i]

    # Tokenize
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the tensors of segments
    # Add [CLS]+[SEP], [MASK]+[SEP] respectively
    segments_ids = [0] * (len(tokenizer.tokenize(emotion_clause)) + 2) + [1] * (len(tokenizer.tokenize(cause_candidate)) + 2)

    # Convert tensors to Pytorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    # Set mode to evaluation
    model.eval()
    model.to(device)

    # Get MASK index
    # Uniconn
    def get_index1(lst=None, item=''):
        return [index for (index,value) in enumerate(lst) if value == item]
    masked_index = get_index1(tokenized_text, '[MASK]')

    # Get prediction
    with torch.no_grad():
        # [1，14，30522] # [#batch, #word, #vocab]
        # Outputs are the probabilities of words
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]  # [1，14，30522] # [#batch, #word, #vocab]

    '''
    # Probability distribution
    print(torch.exp(predictions[0, masked_index])/torch.sum(torch.exp(predictions[0, masked_index])))
    '''

    # Prediction
    candidate_conn_index = []
    candidate_conn_score = []
    for i in range(len(candidate_conn)):
        candidate_conn_index.append(tokenizer.convert_tokens_to_ids(candidate_conn[i]))
        candidate_conn_score.extend(predictions[0, masked_index, candidate_conn_index[i]].cpu().numpy().tolist())
    candidate_index = np.argmax(candidate_conn_score)
    conn = candidate_conn[candidate_index]
    if conn in cause_uniconn:
        is_cause_conn = 'true'
    else:
        is_cause_conn = 'false'

    # Write result in csv
    with open ('../data/pairs_withconn.csv', 'a', encoding='utf-8', newline='') as g:
        csv_writer = csv.writer(g)
        csv_writer.writerow([section, emo_clause_index, cau_candidate_index, emotion_clause, cause_candidate, conn, correctness, is_cause_conn])