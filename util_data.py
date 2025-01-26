import numpy as np
import pandas as pd
import torch
import random


def normalize(adj):
    # 获取对角线的索引
    diagonal_indices = torch.arange(adj.shape[0])
    # 将对角线上的元素设为0
    adj[diagonal_indices, diagonal_indices] = 1
    # 加一个很小的数，避免分母为0
    inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + 1e-10)
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]

# 相似性矩阵的关联网络构建
def k_matrix(matrix, k):
    node_num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(node_num)), axis=1)
    for i in range(node_num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph


def load_data():
    lnc_sim = pd.read_excel(".//dataset//dataset1//08-integrated-lnc.xlsx", header=None).values
    dis_sim = pd.read_excel(".//dataset//dataset1//09-integrated-dis.xlsx", header=None).values
    lnc_dis = pd.read_csv(".//dataset/dataset1/lnc-dis-901-326.csv", header=None).values

    # lnc // dis association matrix
    lnc_inter = np.where(k_matrix(lnc_sim, 15) != 0, 1, k_matrix(lnc_sim, 15))
    dis_inter = np.where(k_matrix(dis_sim, 15) != 0, 1, k_matrix(dis_sim, 15))
    return lnc_sim, lnc_inter, dis_sim, dis_inter, lnc_dis

def load_dataset(args):
    lnc_sim, lnc_inter, dis_sim, dis_inter, lnc_dis = load_data()

    # adj
    heter_net = np.concatenate(
        (np.concatenate((lnc_inter, lnc_dis), axis=1), np.concatenate((lnc_dis.T, dis_inter), axis=1)), axis=0)
    # feature
    heter_feature = np.concatenate(
        (np.concatenate((lnc_sim, lnc_dis), axis=1), np.concatenate((lnc_dis.T, dis_sim), axis=1)), axis=0)

    # return tensor
    return heter_net, torch.FloatTensor(heter_feature)

def edge_list(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.IntTensor(edge_index)

def load_feature(node_feature, graph_feature, args):
    _, _, _, _, asso = load_data()
    zero_index = []
    one_index = []
    for i in range(asso.shape[0]):
        for j in range(asso.shape[1]):
            if asso[i][j] == 0:
                zero = [i, j + asso.shape[0], 0]
                zero_index.append(zero)
            if asso[i][j] == 1:
                one = [i, j + asso.shape[0], 1]
                one_index.append(one)
    lnc_dis_sample = zero_index + one_index

    neg_sample = []
    pos_sample = []

    for ass in lnc_dis_sample:
        if ass[2] == 1:
            pos_sample.append(ass[:2])
        if ass[2] == 0:
            neg_sample.append(ass[:2])

    random.seed(1234)
    random.shuffle(neg_sample)
    neg_sample = random.sample(neg_sample, len(pos_sample))

    sam_node_feature = torch.zeros((len(pos_sample) + len(neg_sample), node_feature.shape[-1] * 2))
    sam_graph_feature = torch.zeros((len(pos_sample) + len(neg_sample), graph_feature.shape[-1] * 2))

    for i in range(len(pos_sample)):
        sam_node_feature[i, :node_feature.shape[-1]] = node_feature[pos_sample[i][0], :]
        sam_node_feature[i, node_feature.shape[-1]:] = node_feature[pos_sample[i][1], :]
        sam_graph_feature[i, :graph_feature.shape[-1]] = graph_feature[pos_sample[i][0], :]
        sam_graph_feature[i, graph_feature.shape[-1]:] = graph_feature[pos_sample[i][1], :]
    for i in range(len(neg_sample)):
        sam_node_feature[len(pos_sample) + i, :node_feature.shape[-1]] = node_feature[neg_sample[i][0], :]
        sam_node_feature[len(pos_sample) + i, node_feature.shape[-1]:] = node_feature[neg_sample[i][1], :]
        sam_graph_feature[len(pos_sample) + i, :graph_feature.shape[-1]] = graph_feature[neg_sample[i][0], :]
        sam_graph_feature[len(pos_sample) + i, graph_feature.shape[-1]:] = graph_feature[neg_sample[i][1], :]

    idx = [i for i in range(len(pos_sample) + len(neg_sample))]

    # label
    labels = []
    for _ in range(len(pos_sample)):
        labels.append([0, 1])
    for _ in range(len(neg_sample)):
        labels.append([1, 0])
    return idx, sam_node_feature, sam_graph_feature, torch.tensor(labels)
