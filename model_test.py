import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BiasedMultiHeadAtten
from torch_geometric.nn.inits import reset
from sklearn.linear_model import LogisticRegression


class MLP(nn.Module): # 两层比较好
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Mish(),
            nn.Linear(out_dim, out_dim),
            nn.Mish(),

        )

        self.fcs1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Mish())


    def forward(self, x):
        return self.fcs(x) + self.fcs1(x) + x

class Model(nn.Module):

    def __init__(self, hidden_channels, encoder1, encoder2, pool, scorer):# , tau=0.5
        super(Model, self).__init__()

        self.hidden_channels = hidden_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 全图encoder+global_mlp
        self.global_mlp = MLP(hidden_channels, hidden_channels).to(self.device)

        self.encoder1 = encoder1 # 子图encoder
        self.encoder2 = encoder2 # 全图encoder

        self.pool = pool
        self.scorer = scorer

        self.marginloss = nn.MarginRankingLoss(0.5, reduction='mean')
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()


    def reset_parameters(self):

        reset(self.encoder1)
        reset(self.encoder2)
        reset(self.pool)
        reset(self.global_mlp)

        reset(self.scorer)


    ## Return node and subgraph representations of each node before and after being shuffled
    def forward(self, x, edge_index, Gdatax, Gdataedge_index, batch=None, index=None):

        hidden = self.encoder1(x, edge_index)
        if index is None:
            return hidden

        # 返回每一个子图的中心节点的嵌入表示
        z = hidden[index]
        # 返回每一个节点对应的子图的嵌入表示
        summary = self.pool(hidden, edge_index, batch)
        # 对全局图进行编码
        Goutput = self.encoder2(Gdatax, Gdataedge_index)

        # Subgraph  Node Embedding
        z = self.global_mlp(z)
        # subgraph graph embedding
        summary = self.global_mlp(summary)
        # Global graph node embedding
        Goutput = self.global_mlp(Goutput)
        return z, summary, Goutput


    def loss(self, hidden1, summary1, Goutput1, lamda):


        ## 针对局部子图进行的loss训练
        # Computes the margin objective.
        # 随机打乱索引
        shuf_index = torch.randperm(summary1.size(0))

        # 打乱后的局部子图节点嵌入
        hidden2 = hidden1[shuf_index]
        # 打乱后的局部子图嵌入
        summary2 = summary1[shuf_index]

        # 样本相似性
        # 原始局部子图和节点之间的相似性
        logits_aa1 = torch.sigmoid(torch.sum(hidden1 * summary1, dim=-1))
        # 打乱后的样本相似性
        logits_bb1 = torch.sigmoid(torch.sum(hidden2 * summary2, dim=-1))
        #
        logits_ab1 = torch.sigmoid(torch.sum(hidden1 * summary2, dim=-1))
        logits_ba1 = torch.sigmoid(torch.sum(hidden2 * summary1, dim=-1))

        TotalLoss1 = 0.0
        # 全1的张量，作为标签
        ones = torch.ones(logits_aa1.size(0)).to(self.device)# 1表示前面一个大，-1表示后面一个大，margin表示相差的值
        TotalLoss1 += self.marginloss(logits_aa1, logits_ba1, ones)
        TotalLoss1 += self.marginloss(logits_bb1, logits_ab1, ones)

        ## 针对全图进行的loss训练
        Goutput2 = Goutput1[shuf_index]
#
        logits_aa2= torch.sigmoid(torch.sum(hidden1 * Goutput1, dim=-1))
        logits_bb2 = torch.sigmoid(torch.sum(hidden2 * Goutput2, dim=-1))
        logits_ab2 = torch.sigmoid(torch.sum(hidden1 * Goutput2, dim=-1))
        logits_ba2 = torch.sigmoid(torch.sum(hidden2 * Goutput1, dim=-1))

        TotalLoss2 = 0.0
        ones = torch.ones(logits_aa2.size(0)).to(self.device)
        TotalLoss2 += self.marginloss(logits_aa2, logits_ba2, ones)
        TotalLoss2 += self.marginloss(logits_bb2, logits_ab2, ones)
# #
        logits_aa3 = torch.sigmoid(torch.sum(summary1 * Goutput1, dim=-1))
        logits_bb3 = torch.sigmoid(torch.sum(summary2 * Goutput2, dim=-1))
        logits_ab3 = torch.sigmoid(torch.sum(summary1 * Goutput2, dim=-1))
        logits_ba3 = torch.sigmoid(torch.sum(summary2 * Goutput1, dim=-1))

        TotalLoss3 = 0.0
        ones = torch.ones(logits_aa1.size(0)).to(self.device)
        TotalLoss3 += self.marginloss(logits_aa3, logits_ba3, ones)
        TotalLoss3 += self.marginloss(logits_bb3, logits_ab3, ones)

        TotalLoss = lamda * TotalLoss1 + (TotalLoss2 + TotalLoss3) / 2

        return TotalLoss

        # return ret


class TransClf(nn.Module):
    def __init__(self, hidden_size, num_heads, atten_bias_method='add', norm_first=False, dropout=0.1, atten_dropout=0.1):
        super(TransClf, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.norm_first = norm_first
        self.atten = BiasedMultiHeadAtten(hidden_size, num_heads)
        self.lin = nn.Linear(hidden_size * 4, hidden_size).to(self.device)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Mish(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Mish()
        )
        self.dropout = nn.Dropout(dropout)
        self.atten_layer_norm = nn.LayerNorm(hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        self.prediction = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Mish(),
            nn.Linear(hidden_size // 2, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, embedding, atten_bias=None):
        residual = self.lin(embedding.to(self.device))
        if self.norm_first:
            embedding = self.atten_layer_norm(embedding)
        # graph bias
        embedding = residual + self.dropout(self.atten(embedding, atten_bias))

        if not self.norm_first:
            embedding = self.atten_layer_norm(embedding)
        if self.norm_first:
            embedding = self.ffn_layer_norm(embedding)
        embedding = residual + self.ffn(embedding)
        if not self.norm_first:
            embedding = self.ffn_layer_norm(embedding)
        # prediction
        return self.prediction(embedding)







