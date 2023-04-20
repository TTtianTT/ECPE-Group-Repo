import os
import csv
from config import *
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification
from model import Network
import datetime
import numpy as np
import pandas as pd
import pickle
from utils.utils import *
from accelerate import Accelerator


'''
batch size = 1 for experiment
'''


# dataset
class Discourse(Dataset):
    def __init__(self, tokenizer, path):
        self.tokenizer = tokenizer
        self.data_path = path
        self.discourses_list = []
        df = pd.read_csv(self.data_path)
        for i in range(len(df)):
            section = int(df['section'][i])
            discourse = torch.Tensor(self.tokenizer(df['discourse'][i],padding='max_length',max_length=512)['input_ids']).to(torch.int32)
            word_count = int(df['word_count'][i])
            doc_len = int(df['doc_len'][i])
            clause_len = df['clause_len'][i]
            emotion_pos = df['emotion_pos'][i]
            cause_pos = df['cause_pos'][i]
            true_pairs = df['true_pairs'][i]
            conn = torch.Tensor(self.tokenizer(df['conn'][i],padding='max_length',max_length=512)['input_ids']).to(torch.int32)
            self.discourses_list.append([section, discourse, word_count, doc_len, clause_len, emotion_pos, cause_pos, true_pairs, conn])

    def __getitem__(self, item):
        item = self.discourses_list[item]
        return item

    def __len__(self):
        return len(self.discourses_list)


# evaluate one batch
def evaluate_one_batch(configs, batch, model, tokenizer):
    # 1 doc has 3 emotion clauses and 4 cause clauses at most, respectively
    # 1 emotion clause has 3 corresponding cause clauses at most, 1 cause clause has only 1 corresponding emotion clause
    with open('data/sentimental_clauses.pkl', 'rb') as f:
        emo_dictionary = pickle.load(f)

    section, discourse, word_count, doc_len, clause_len, emotion_pos, cause_pos, true_pairs, conn = batch

    emotion_pos = eval(emotion_pos[0])
    cause_pos = eval(cause_pos[0])
    true_pairs = eval(true_pairs[0])
    discourse_mask = torch.Tensor([1] * word_count + [0] * (512 - word_count)).to(torch.int32)
    segment_mask = torch.Tensor([0] * 512).to(torch.int32)

    query_len = 0
    discourse_adj = torch.ones([doc_len, doc_len])  # batch size = 1
    emo_ans = torch.zeros(doc_len)
    for pos in emotion_pos:
        emo_ans[int(pos) - 1] = 1
    emo_ans_mask = torch.ones(doc_len)  # batch size = 1

    pair_count = len(emotion_pos) * (doc_len - len(emotion_pos) + 1)

    emo_cau_ans = torch.zeros(len(emotion_pos) * doc_len)
    for i in range(len(emotion_pos)):
        for j in range(len(cause_pos[i])):
            emo_cau_ans[int(doc_len) * i + cause_pos[i][j] - 1] = 1
    emo_cau_ans_mask = torch.ones(len(emotion_pos) * doc_len)

    # due to batch = 1, dim0 = 1
    true_emo = emotion_pos
    true_cau = []
    for emo in cause_pos:
        for cau in emo:
            true_cau.append(cau)

    pred_emo_f = []
    pred_pair_f = []
    pred_pair_f_pro = []
    pred_emo_single = []
    pred_cau_single = []
    
    section = str(section.item())

    # step 1
    f_emo_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, emotion_pos, cause_pos, doc_len, discourse_adj, conn, 'emo')
    emo_ans_mask = emo_ans_mask.to(DEVICE)
    temp_emo_f_prob = f_emo_pred.masked_select(emo_ans_mask.bool()).cpu().numpy().tolist()
    for idx in range(len(temp_emo_f_prob)):
        if temp_emo_f_prob[idx] > 0.99 or (temp_emo_f_prob[idx] > 0.5 and idx + 1 in emo_dictionary[section]):
            pred_emo_f.append(idx)
            pred_emo_single.append(idx + 1)

    # step 2
    for idx_emo in pred_emo_f:
        f_cau_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, emotion_pos, cause_pos, doc_len, discourse_adj, conn, 'emo_cau')  
        temp_cau_f_prob = f_cau_pred[0].cpu().numpy().tolist()

        for idx_cau in range(len(temp_cau_f_prob)):
            if temp_cau_f_prob[idx_cau] > 0.5 and abs(idx_emo - idx_cau) <= 11:
                if idx_cau + 1 not in pred_cau_single:
                    pred_cau_single.append(idx_cau + 1)
                prob_t = temp_emo_f_prob[idx_emo] * temp_cau_f_prob[idx_cau]
                if idx_cau - idx_emo >= 0 and idx_cau - idx_emo <= 2:
                    pass
                else:
                    prob_t *= 0.9
                pred_pair_f_pro.append(prob_t)
                pred_pair_f.append([idx_emo + 1, idx_cau + 1])

    pred_emo_final = []
    pred_cau_final = []
    pred_pair_final = []

    for i, pair in enumerate(pred_pair_f):
        if pred_pair_f_pro[i] > 0.5:
            pred_pair_final.append(pair)

    for pair in pred_pair_final:
        if pair[0] not in pred_emo_final:
            pred_emo_final.append(pair[0])
        if pair[1] not in pred_cau_final:
            pred_cau_final.append(pair[1])

    metric_e, metric_c, metric_p = \
        cal_metric(pred_emo_final, true_emo, pred_cau_final, true_cau, pred_pair_final, true_pairs, doc_len)
    return metric_e, metric_c, metric_p


# evaluate step
def evaluate(configs, test_loader, model, tokenizer):
    model.eval()
    all_emo, all_cau, all_pair = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    for batch in test_loader:
        emo, cau, pair = evaluate_one_batch(configs, batch, model, tokenizer)
        for i in range(3):
            all_emo[i] += emo[i]
            all_cau[i] += cau[i]
            all_pair[i] += pair[i]

    eval_emo = eval_func(all_emo)
    eval_cau = eval_func(all_cau)
    eval_pair = eval_func(all_pair)
    return eval_emo, eval_cau, eval_pair


def main(configs, train_loader, test_loader, tokenizer):
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
    training_steps = configs.epochs * len(train_loader) // configs.gradient_accumulation_steps
    warmup_steps = int(training_steps * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    # training
    model.zero_grad()
    max_result_pair, max_result_emo, max_result_cau = None, None, None
    max_result_emos, max_result_caus = None, None
    early_stop_flag = 0

    for epoch in range(1, configs.epochs+1):
        for train_step, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            
            section, discourse, word_count, doc_len, clause_len, emotion_pos, cause_pos, true_pairs, conn = batch

            emotion_pos = eval(emotion_pos[0])
            cause_pos = eval(cause_pos[0])
            discourse_mask = torch.Tensor([1] * word_count + [0] * (512 - word_count)).to(torch.int32)
            segment_mask = torch.Tensor([0] * 512).to(torch.int32)

            query_len = 0
            discourse_adj = torch.ones([doc_len, doc_len])  # batch size = 1
            emo_ans = torch.zeros(doc_len)
            for pos in emotion_pos:
                emo_ans[int(pos) - 1] = 1
            emo_ans_mask = torch.ones(doc_len)  # batch size = 1

            pair_count = len(emotion_pos) * (doc_len - len(emotion_pos) + 1)

            emo_cau_ans = torch.zeros(len(emotion_pos) * doc_len)
            for i in range(len(emotion_pos)):
                for j in range(len(cause_pos[i])):
                    emo_cau_ans[int(doc_len) * i + cause_pos[i][j] - 1] = 1
            emo_cau_ans_mask = torch.ones(len(emotion_pos) * doc_len)

            emo_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, emotion_pos, cause_pos, doc_len, discourse_adj, conn, 'emo')
            emo_cau_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, emotion_pos, cause_pos, doc_len, discourse_adj, conn, 'emo_cau')
            
            loss_emo = model.loss_pre_emo(emo_pred, emo_ans, emo_ans_mask)
            loss_emo_cau = model.loss_pre_emo_cau(emo_cau_pred, emo_cau_ans, emo_cau_ans_mask)
            
            loss = loss_emo + loss_emo_cau
            loss.backward()

            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if train_step % 1 == 0:
                print('epoch: {}, step: {}, loss_emo: {}, loss_emo_cau: {}, loss: {}'
                    .format(epoch, train_step, loss_emo, loss_emo_cau, loss))
        
        with torch.no_grad():
            eval_emo, eval_cau, eval_pair = evaluate(configs, test_loader, model, tokenizer)
            
            if max_result_pair is None or eval_pair[0] > max_result_pair[0]:
                early_stomax_result_pairp_flag = 1
                max_result_emo = eval_emo
                max_result_cau = eval_cau
                max_result_pair = eval_pair
    
                state_dict = {'model': model.state_dict(), 'result': max_result_pair}
                # torch.save(state_dict, 'model/model.pth')
            else:
                early_stop_flag += 1
        if epoch > configs.epochs / 2 and early_stop_flag >= 7:
            break

    return max_result_emo, max_result_cau, max_result_pair


if __name__ == '__main__':
    configs = Config()
    device = DEVICE
    tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
    model = Network(configs).to(DEVICE)
    
    train_dataset = Discourse(tokenizer, configs.train_dataset_path)
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1,drop_last=True)
    test_dataset = Discourse(tokenizer, configs.test_dataset_path)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)
    
    max_result_emo, max_result_cau, max_result_pair = main(configs, train_loader, test_loader, tokenizer)
    print('max_result_emo: {}, max_result_cau: {}, max_result_pair: {}'.format(max_result_emo, max_result_cau, max_result_pair))