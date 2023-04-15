import csv
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM

# Init param
cause_conn = []
contrast_conn = []
conjunction_conn = []
condition_conn = []
restatement_conn = []
candidate_conn = []
line_count = 0
with open ('../data/specified_conn.txt', 'r', encoding='utf-8') as f:
    line_count += 1
    line = f.readline()
    while line:
        for word in line.rstrip().split(' ')[1:]:
            if line_count == 1:
                cause_conn.append(word)
            if line_count == 2:
                contrast_conn.append(word)
            if line_count == 3:
                conjunction_conn.append(word)
            if line_count == 4:
                condition_conn.append(word)
            if line_count == 5:
                restatement_conn.append(word)
            candidate_conn.append(word)
        line_count += 1
        line = f.readline()

# Load dataset
df = pd.read_csv('../data/test/pairs.csv')

# Init csv of result
with open ('../data/test/pairs_withconn&possibility.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['section', 'emo_clause_index', 'cau_candidate_index', 'emotion_clause', 'cause_candidate', 'conn_words', 'possibility_distribution', 'correctness', 'cause_conn_possibility', 'contrast_conn_possibility', 'conjunction_conn_possibility', 'condition_conn_possibility', 'restatement_conn_possibility'])

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
    possibility_distribution = np.exp(candidate_conn_score)/np.sum(np.exp(candidate_conn_score))
    cause_conn_possibility, contrast_conn_possibility, conjunction_conn_possibility, condition_conn_possibility, restatement_conn_possibility = np.sum(possibility_distribution[0:4]), np.sum(possibility_distribution[4:8]), np.sum(possibility_distribution[8:12]), np.sum(possibility_distribution[12:16]), np.sum(possibility_distribution[16:20])

    # Write result in csv
    with open ('../data/test/pairs_withconn&possibility.csv', 'a', encoding='utf-8', newline='') as g:
        csv_writer = csv.writer(g)
        csv_writer.writerow([section, emo_clause_index, cau_candidate_index, emotion_clause, cause_candidate, candidate_conn, possibility_distribution, correctness, cause_conn_possibility, contrast_conn_possibility, conjunction_conn_possibility, condition_conn_possibility, restatement_conn_possibility])