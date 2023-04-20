from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from config import DEVICE
from gnn_layer import GraphAttentionLayer


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert_encoder = BertEncoder(configs)
        self.gnn = GraphNN(configs)
        self.pred_emo = Pre_Predictions_emo(configs)
        self.pred_emo_cau = Pre_Predictions_emo_cau(configs)

    def forward(self, query, query_mask, query_seg, query_len, clause_len, emotion_pos, cause_pos, doc_len, adj, conn, q_type):
        # shape: batch_size, max_doc_len, 1024
        doc_sents_h = self.bert_encoder(query, query_mask, query_seg, query_len, clause_len, doc_len)
        doc_sents_h = self.gnn(doc_sents_h, doc_len, adj)
        pred_emo = self.pred_emo(doc_sents_h)
        pred_emo_cau = self.pred_emo_cau(doc_sents_h, emotion_pos, doc_len, conn)
        if q_type == 'emo':
            return pred_emo
        if q_type == 'emo_cau':
            return pred_emo_cau
        return None

    def loss_pre_emo(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(DEVICE)  # shape: batch_size, seq_len
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        # weight = torch.where(true > 0.5, 2, 1)
        criterion = nn.BCELoss()
        return criterion(pred, true)

    def loss_pre_emo_cau(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(DEVICE)  # shape: batch_size, seq_len
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        # weight = torch.where(true > 0.5, 2, 1)
        criterion = nn.BCELoss()
        return criterion(pred, true)


class BertEncoder(nn.Module):
    def __init__(self, configs):
        super(BertEncoder, self).__init__()
        hidden_size = configs.feat_dim
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.fc = nn.Linear(768, 1)

    def forward(self, discourse, discourse_mask, segment_mask, query_len, clause_len, doc_len):
        hidden_states = self.bert(input_ids=discourse.to(DEVICE),
                                  attention_mask=discourse_mask.to(DEVICE),
                                  token_type_ids=segment_mask.to(DEVICE))[0]
        hidden_states, mask_doc = self.get_sentence_state(hidden_states, query_len, clause_len, doc_len)

        alpha = self.fc(hidden_states).squeeze(-1)  # bs, max_doc_len, max_seq_len
        mask_doc = 1 - mask_doc # bs, max_doc_len, max_seq_len
        alpha.data.masked_fill_(mask_doc.bool(), -9e5)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(-1).repeat(1, 1, 1, hidden_states.size(-1))
        hidden_states = torch.sum(alpha * hidden_states, dim=2) # bs, max_doc_len, 768

        return hidden_states.to(DEVICE)

    def get_sentence_state(self, hidden_states, query_lens, clause_lens, doc_len):
        # 对文档的每个句子的token做注意力，得到每个句子的向量表示
        sentence_state_all = []
        mask_all = []
        max_clause_len = 0
        clause_lens = eval(clause_lens[0])
        clause_lens = [clause_lens]

        for clause_len in clause_lens: # 找出最长的一句话包含多少token
            for l in clause_len:
                max_clause_len = max(max_clause_len, l)

        max_doc_len = max(doc_len) # 最长的文档包含多少句子
        for i in range(hidden_states.size(0)):  # 对每个batch
            # 对文档sentence
            mask = []
            begin = 0
            sentence_state = []
            for clause_len in clause_lens[i]:
                sentence = hidden_states[i, begin: begin + clause_len]
                begin += clause_len
                if sentence.size(0) < max_clause_len:
                    sentence = torch.cat([sentence, torch.zeros((max_clause_len - clause_len, sentence.size(-1))).to(DEVICE)],
                                         dim=0)
                sentence_state.append(sentence.unsqueeze(0))
                mask.append([1] * clause_len + [0] * (max_clause_len - clause_len))
            sentence_state = torch.cat(sentence_state, dim=0).to(DEVICE)
            if sentence_state.size(0) < max_doc_len:
                mask.extend([[0] * max_clause_len] * (max_doc_len - sentence_state.size(0)))
                padding = torch.zeros(
                    (max_doc_len - sentence_state.size(0), sentence_state.size(-2), sentence_state.size(-1)))
                sentence_state = torch.cat([sentence_state, padding.to(DEVICE)], dim=0)
            sentence_state_all.append(sentence_state.unsqueeze(0))
            mask_all.append(mask)
        sentence_state_all = torch.cat(sentence_state_all, dim=0).to(DEVICE)
        mask_all = torch.tensor(mask_all).to(DEVICE)
        return sentence_state_all, mask_all


class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]  # [1024, 256]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')] # [4]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len
        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)
        return doc_sents_h

class Pre_Predictions_emo(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions_emo, self).__init__()
        self.feat_dim = 768
        self.out_emo = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_emo = self.out_emo(doc_sents_h).squeeze(-1)  # bs, max_doc_len, 1
        pred_emo = torch.sigmoid(pred_emo)
        return pred_emo # shape: bs ,max_doc_len

class Pre_Predictions_emo_cau(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions_emo_cau, self).__init__()
        self.feat_dim = 768
        self.out_emo_cau = nn.Linear(self.feat_dim, 1)
        self.linear_layer = nn.Linear(3, 1)
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
 
    def forward(self, doc_sents_h, emotion_pos, doc_len, conn):
        doc_sents_h_2d = doc_sents_h.squeeze(0)  # shape: batch_size=1, max_doc_len, 768, squeee dim0
        # Set single mask for one connective
        mask = torch.tensor([1] + [0] * 511).unsqueeze(0)
        segement = torch.tensor([0] * 512).unsqueeze(0)
        # Init pairs_h
        pairs_h = torch.tensor([]).to(DEVICE)
        for i in range(len(emotion_pos)):
            for j in range(doc_len):
                pos = i * doc_len + j
                inputs = conn[0][pos]
                inputs = F.pad(inputs, (0, 512 - inputs.size(-1)), 'constant', 0).unsqueeze(0)
                
                # Get connective embedding
                conn_embedding = self.bert(inputs.to(DEVICE), mask.to(DEVICE), segement.to(DEVICE))[0][0][0]
                
                # Stack three embeddings for one pair presentation
                pair_h = torch.stack([doc_sents_h_2d[emotion_pos[i] - 1], conn_embedding, doc_sents_h_2d[j]], dim=-1)
                pair_h = self.linear_layer(pair_h).unsqueeze(-1)
                
                # Concatenate pairs for whole doc answer
                if pairs_h == torch.Size([]):
                    pairs_h = pair_h
                else:
                    pairs_h = torch.concatenate([pairs_h, pair_h], dim=-1)
        
        pairs_h = torch.permute(pairs_h, (1,2,0))
        pred_emo_cau = self.out_emo_cau(pairs_h)  # bs, max_doc_len, 1
        pred_emo_cau = pred_emo_cau.squeeze(-1)
        pred_emo_cau = torch.sigmoid(pred_emo_cau)
        return pred_emo_cau # shape: bs , emo_num * max_doc_len
