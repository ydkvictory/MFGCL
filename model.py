import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling, GATConv

# GCN + GAT
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_channels = hidden_channels
        self.conv = GCNConv(in_channels, self.hidden_channels).to(self.device)
        self.lin = nn.Linear(in_channels, hidden_channels).to(self.device)
        self.mish = nn.Mish()


    def forward(self, x, edge_index):
        x1 = self.conv(x.to(self.device), edge_index.to(self.device))
        x1 = self.mish(x1)
        return x1

class Pool(nn.Module):
    def __init__(self, in_channels, ratio=1.0):
        super(Pool, self).__init__()
        # 自注意力图池化
        self.sag_pool = SAGPooling(in_channels, ratio)
        self.lin1 = torch.nn.Linear(in_channels * 2, in_channels)

    def forward(self, x, edge, batch, type='soft_pool'):
        # 通过batch了解每个节点属于哪个子图，然后对对应的信息进行分组，并pool
        if type == 'mean_pool':
            return global_mean_pool(x, batch)
        elif type == 'max_pool':
            return global_max_pool(x, batch)
        elif type == 'sum_pool':
            return global_add_pool(x, batch)
        elif type == 'sag_pool':
            x1, _, _, batch, _, _ = self.sag_pool(x, edge, batch=batch)
            return global_mean_pool(x1, batch)
        elif type == 'soft_pool':
            # first torch.exp(x) and element-wise multiply , and mean_pool
            # return global_mean_pool(torch.softmax(x, dim=1), batch)
            w = torch.exp(x)
            return global_mean_pool(x.mul(w), batch).div_(global_mean_pool(w, batch))


class Scorer(nn.Module):
    def __init__(self, hidden_size):
        super(Scorer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

    def forward(self, input1, input2):
        output = torch.sigmoid(torch.sum(input1 * torch.matmul(input2, self.weight), dim=-1))
        return output


class BiasedMultiHeadAtten(nn.Module):
    def __init__(self, hidden_size, num_head, bias=True, atten_bias_method="add", atten_drop=0.1):
        super(BiasedMultiHeadAtten, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.head_dim = self.hidden_size // self.num_head
        assert (self.head_dim * self.num_head == self.hidden_size), 'dimension of embedding must be divisible by number of head'
        self.scale = self.head_dim ** -0.5
        self.atten_bias_method = atten_bias_method

        self.query = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=bias)
        self.key = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=bias)
        self.vaule = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=bias)
        self.res = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.lin = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=bias)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.dropout = nn.Dropout(atten_drop)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.query.weight, gain= 2 ** -0.5)
        nn.init.xavier_uniform_(self.key.weight, gain= 2 ** -0.5)
        nn.init.xavier_uniform_(self.vaule.weight, gain= 2 ** -0.5)
        nn.init.xavier_uniform_(self.res.weight, gain= 2 ** -0.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, node_embedding, atten_bias=None):
        query = self.query(node_embedding).transpose(0, 1)
        key = self.key(node_embedding).transpose(0, 1)
        value = self.key(node_embedding).transpose(0, 1)

        res = self.lin(atten_bias)
        res = res * torch.sigmoid(self.res(res)).unsqueeze(0)

        # batch = 1, len = batch, _ = dim
        batch, len, _ = node_embedding.unsqueeze(0).shape
        query = query.reshape(len, batch * self.num_head, self.head_dim).transpose(0, 1) * self.scale
        key = key.reshape(len, batch * self.num_head, self.head_dim).permute(1, 2, 0)
        value = value.reshape(len, batch * self.num_head, self.head_dim).transpose(0, 1)
        # bmm is batch matrix multiply
        atten = torch.bmm(query, key).transpose(0, 2).reshape(len, -1, batch, self.num_head).transpose(0, 2)

        atten = self.dropout(torch.softmax(atten.transpose(0, 2).reshape(len, -1, batch * self.num_head).transpose(0,2), dim=2))

        atten = torch.bmm(atten, value).transpose(0, 1)
        atten = (self.out_proj(atten.reshape(len, batch, self.hidden_size).transpose(0, 1)) + res).squeeze()
        return atten
