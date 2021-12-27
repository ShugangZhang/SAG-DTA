# gat.py
# This is a file from GraphDTA without modification.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj

from utils import *
import torch
from torch_geometric.utils import add_self_loops, degree


class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        #self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        # 1.attention
        h = self.W(x)
        batch_size = h.size()[0]
        N = h.size()[1]
        e = torch.einsum('il,jl->ij', (torch.matmul(h, self.A), h))
        e = e + e.permute((1, 0))
        zero_vec = -9e15*torch.ones_like(e)
        # obtain adjacency matrix according to edge_index
        adj = torch.zeros_like(e)
        edge_index_trans = edge_index.transpose(1,0)
        for idx in edge_index_trans:
            i = idx[0]
            j = idx[1]
            adj[i][j] = 1
        for i in range(len(adj[0])):
            adj[i][i] = 1
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        #h_prime = torch.matmul(attention, h)
        attention = attention*adj  # distance aware
        h_prime = F.relu(torch.einsum('ij,jk->ik', (attention, h)))

        # 2.gate
        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1)))
        retval = coeff*x+(1-coeff)*h_prime
        return retval


class MyGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(MyGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, edge_index):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # obtain adjacency matrix according to edge_index
        adj = torch.zeros_like(e)
        edge_index_trans = edge_index.transpose(1,0)
        for idx in edge_index_trans:
            i = idx[0]
            j = idx[1]
            adj[i][j] = 1
        for i in range(len(adj[0])):
            adj[i][i] = 1

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# 这个是重点，包括pooling ratio是怎么用的，都得关注
class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    # edge_index从何处来？x又是什么？明天，带着希望继续。20200506
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        # squeeze的意思是从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        # print("x.shape ===== ", x.shape)
        score = self.score_layer(x, edge_index).squeeze()  # 每个node的评分  # 有无经历softmax？
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]  # 确实是按照传递进来的ratio在不断pooling的
        edge_index, edge_attr = filter_adj(  # 模糊能感觉出这个是想处理删点之后的邻接关系
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm