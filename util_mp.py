import os
import torch
import numpy as np
from cytoolz import curry
import multiprocessing as mp
from scipy import sparse as sp
from sklearn.preprocessing import normalize, StandardScaler
from torch_geometric.data import Data, Batch


def standardize(feat, mask):
    scaler = StandardScaler()
    scaler.fit(feat[mask])
    new_feat = torch.FloatTensor(scaler.transform(feat))
    return new_feat


def preprocess(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.tensor(features)


## 计算邻居节点的重要性，并打分数
class PPR:
    # Node-wise personalized pagerank
    def __init__(self, adj_mat, maxsize=20, n_order=10, alpha=0.85):  # maxsize=20
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()

    def search(self, seed, alpha=0.85):
        # 创建一个shape形状的矩阵，行为seed，列为0的位置为1
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        # 阶数 hop 游走的跳数
        # ppr
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)

        idx = scores.argsort()[::-1][:self.maxsize]  # 返回分数最高的节点的索引
        # 得分最高的列索引
        neighbor = np.array(x.indices[idx])

        # 找到种子节点的行索引
        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            # 如果种子节点不在邻居节点中，将种子节点放在最前面
            neighbor = np.append(np.array([seed]), neighbor)
        else:
            seed_idx = seed_idx[0]
            # 在，将种子节点和第一个节点互换
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]

        # 保证有一个种子节点
        assert np.where(neighbor == seed)[0].size == 1
        # 保证种子节点在第一个
        assert np.where(neighbor == seed)[0][0] == 0

        return neighbor

    @curry
    def process(self, path, seed):
        ppr_path = os.path.join(path, 'ppr{}'.format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            print('Processing node {}.'.format(seed))
            # 得到种子节点的邻居节点
            neighbor = self.search(seed)
            torch.save(neighbor, ppr_path)
        else:
            print('File of node {} exists.'.format(seed))

    def search_all(self, node_num, path):
        neighbor = {}
        if os.path.isfile(path + '_neighbor') and os.stat(path + '_neighbor').st_size != 0:
            print("Exists neighbor file")
            neighbor = torch.load(path + '_neighbor')
        else:
            print("Extracting subgraphs")
            os.system('mkdir {}'.format(path))
            with mp.Pool() as pool:
                # 并行处理所有节点的邻居节点
                list(pool.imap_unordered(self.process(path), list(range(node_num)), chunksize=1000))

            print("Finish Extracting")
            for i in range(node_num):
                # 将每一个节点的邻居节点写入neighbor
                neighbor[i] = torch.load(os.path.join(path, 'ppr{}'.format(i)))
            torch.save(neighbor, path + '_neighbor')
            os.system('rm -r {}'.format(path))
            print("Finish Writing")
        return neighbor


class Subgraph:
    # Class for subgraph extraction
    def __init__(self, x, edge_index, path, maxsize, n_order):
        self.x = x
        self.path = path
        self.edge_index = np.array(edge_index)
        self.edge_num = edge_index[0].size(0)
        self.node_num = x.size(0)
        self.maxsize = maxsize

        # 创建一个稀疏矩阵，有边设为1，无边设为0
        self.sp_adj = sp.csc_matrix((np.ones(self.edge_num), (edge_index[0], edge_index[1])),
                                    shape=[self.node_num, self.node_num])
        self.ppr = PPR(self.sp_adj, n_order=n_order)

        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}

    def process_adj_list(self):
        # 为每个节点创建一个集合
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            # 得到当前边的两个节点
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            # 将两个节点相互添加到对方的集合中，无向图
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)
        # 可以得到每个节点的邻居节点

    def adjust_edge(self, idx):
        # Generate edges for subgraphs
        dic = {}
        for i in range(len(idx)):
            # 对于选定节点，重新编码
            dic[idx[i]] = i

        new_index = [[], []]
        # 包含选定index的集合
        nodes = set(idx)
        for i in idx:
            # 将节点i和邻居节点和node求交集
            edge = list(self.adj_list[i] & nodes)
            # 得到重新编码后的边索引
            edge = [dic[_] for _ in edge]
            # edge = [_ for _ in edge if _ > i]
            # 选定节点相邻的边的起始节点
            new_index[0] += len(edge) * [dic[i]]
            # 目标节点
            new_index[1] += edge
        return torch.LongTensor(new_index)

    def adjust_x(self, idx):
        # Generate node features for subgraphs
        return self.x[idx]

    def build(self):
        # Extract subgraphs for all nodes
        if os.path.isfile(self.path + '_subgraph') and os.stat(self.path + '_subgraph').st_size != 0:
            print("Exists subgraph file")
            self.subgraph = torch.load(self.path + '\_subgraph')
            return

        self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            # 每个节点的top-k节点
            nodes = self.neighbor[i][:self.maxsize]

            x = self.adjust_x(nodes)
            edge = self.adjust_edge(nodes)
            # 将每个节点的邻居图和属性保存在subgraph
            self.subgraph[i] = Data(x, edge)
        torch.save(self.subgraph, self.path + '\_subgraph')

    def search(self, node_list):
        # Extract subgraphs for nodes in the list
        batch = []
        index = []
        size = 0
        for node in node_list:
            # 取出属性和邻居
            # batch包含每个节点对应的邻居子图以及属性
            batch.append(self.subgraph[node])
            # index记录每个节点在batch列表中的起始索引
            index.append(size)
            # 节点的结束索引
            size += self.subgraph[node].x.size(0)
        # index 这个简直就是神来之笔，妙得很！
        # index记录了中心节点的索引
        index = torch.tensor(index)
        # 将列表转化为batch批次对象
        batch = Batch().from_data_list(batch)
        return batch, index








