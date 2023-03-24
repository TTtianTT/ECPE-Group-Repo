# gnn的整体结构文件
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
      
      
# gnn层具体实现的函数文件     
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from config import DEVICE


class GraphAttentionLayer(nn.Module):
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)

    def forward(self, feat_in, adj=None):
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim

        feat_in_ = feat_in.unsqueeze(1)
        h = torch.matmul(feat_in_, self.W)

        attn_src = torch.matmul(torch.tanh(h), self.w_src)
        attn_dst = torch.matmul(torch.tanh(h), self.w_dst)
        attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)

        adj = torch.FloatTensor(adj).to(DEVICE)
        mask = 1 - adj.unsqueeze(1)
        attn.data.masked_fill_(mask.bool(), -999)

        attn = F.softmax(attn, dim=-1)
        feat_out = torch.matmul(attn, h) + self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = torch.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
