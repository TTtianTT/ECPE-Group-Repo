import os
import csv
from config import *
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification
from GNNmodel import Network
import datetime
import numpy as np
import pandas as pd

'''
discourse开头加上cls
'''


class Discourse(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_path = "discourse.csv"
        self.discourses_list = []
        df = pd.read_csv(self.data_path)
        for i in range(len(df)):
            section = int(df['section'][i])
            discourse = df['discourse'][i]
            word_count = int(df['word_count'][i])
            doc_len = int(df['doc_len'][i])
            clause_len = df['clause_len'][i]
            emotion_pos = df['emotion_pos'][i]
            cause_pos = df['cause_pos'][i]
            self.discourses_list.append([section, discourse, word_count, doc_len, clause_len, emotion_pos, cause_pos])

    def __getitem__(self, item):
        # line = self.discourses_list[item]
        # encoding = self.tokenizer(line[1], return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        # return {'input_ids': encoding['input_ids'].squeeze(),
        #         'token_type_ids': encoding['token_type_ids'].squeeze(),
        #         'attention_mask': encoding['attention_mask'].squeeze(),
        #         }
        item = self.discourses_list[item]
        item[1] = torch.Tensor(self.tokenizer(item[1],padding='max_length',max_length=512)['input_ids']).to(torch.int32)
        # print(item[1])
        return item

    def __len__(self):
        return len(self.discourses_list)


class PairsWithConnectivesDataSet(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_path = "pairs_withconn&possibility.csv"
        self.pairs_list = []
        df = pd.read_csv(self.data_path)
        for i in range(len(df)):
            section = df['section'][i]
            emo_clause_index = df['emo_clause_index'][i]
            cau_candidate_index = df['cau_candidate_index'][i]
            emotion_clause = df['emotion_clause'][i]
            cause_candidate = df['cause_candidate'][i]
            conn_words = df['conn_words'][i]
            possibility_distribution = df['possibility_distribution'][i]
            correctness = df['correctness'][i]
            self.discourses_list.append(
                [section, emo_clause_index, cau_candidate_index, emotion_clause, cause_candidate, conn_words,
                 possibility_distribution, correctness])

    def __getitem__(self, item):
        line = self.pairs_list[item]
        correctness = 0 if line[7] == 'False' else 1
        encoding = self.tokenizer(line[3], line[5] + ' ' + line[4], return_tensors='pt', padding='max_length',
                                  truncation=True, max_length=512)
        return {'input_ids': encoding['input_ids'].squeeze(),
                'token_type_ids': encoding['token_type_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'correctness': correctness,
                }

    def __len__(self):
        return len(self.pairs_list)


def main(configs, dataloader, tokenizer):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    # model
    model = Network(configs).to(DEVICE)
    # optimizer
    params = list(model.named_parameters())
    optimizer_grouped_params = [
        {'params': [p for n, p in params if '_bert' in n], 'weight_decay': 0.01},
        {'params': [p for n, p in params if '_bert' not in n], 'lr': configs.lr, 'weight_decay': 0.01}
    ]
    optimizer = AdamW(params=optimizer_grouped_params, lr=configs.tuning_bert_rate)

    # scheduler
    training_steps = configs.epochs * len(dataloader) // configs.gradient_accumulation_steps
    warmup_steps = int(training_steps * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    # training
    model.zero_grad()
    early_stop_flag = None

    for epoch in range(1, configs.epochs + 1):
        for train_step, batch in enumerate(dataloader):
            model.train()

            section, discourse, word_count, doc_len, clause_len, emotion_pos, cause_pos = batch


            emotion_pos = eval(emotion_pos[0])
            discourse_mask = torch.Tensor([1] * word_count + [0] * (512 - word_count)).to(torch.int32)
            # segment_mask = torch.Tensor([0] * word_count)
            segment_mask = torch.Tensor([0] * 512).to(torch.int32)

            query_len = 0
            # discourse_adj = torch.ones([doc_len, doc_len]).unsqueeze(0)  # batch size = 1
            discourse_adj = torch.ones([doc_len, doc_len])  # batch size = 1

            emo_ans = torch.nn.functional.one_hot(torch.tensor([int(pos) - 1 for pos in emotion_pos]),
                                                  num_classes=doc_len.item())
            emo_ans_mask = torch.ones(doc_len)  # batch size = 1
            # print(discourse_mask.unsqueeze(0))
            print(discourse.size())
            print(discourse_mask.unsqueeze(0))
            emo_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, doc_len, discourse_adj,
                             'f_emo')

            loss_emo = model.loss_pre(emo_pred, emo_ans, emo_ans_mask)
            loss_emo.backward()

            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if train_step % 1 == 0:
                print('epoch: {}, step: {}, loss: {}'.format(epoch, train_step, loss_emo))

    return None


if __name__ == '__main__':
    # preprocess()
    # Bert_conn_possibility()
    configs = Config()
    device = DEVICE
    tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
    model = Network(configs).to(DEVICE)
    discourse_dataset = Discourse(tokenizer)
    discourse_dataloader = DataLoader(dataset=discourse_dataset, shuffle=False, batch_size=1)

    main(configs, discourse_dataloader, tokenizer)