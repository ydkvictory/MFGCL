import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_mp import Subgraph
from model_test import Model, TransClf
from model import Pool, Scorer, Encoder
from util_data import load_dataset, edge_list, load_feature
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch.utils import data
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score, \
    confusion_matrix, recall_score, precision_score, f1_score, roc_curve, auc, precision_recall_curve, matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_parser():
    parser = argparse.ArgumentParser(description='Description : My GCL Model')
    parser.add_argument('--dataset', help='dataset1(901-326), dataset2(356-316), dataset3(240-412), default=dataset1', default='dataset1')
    parser.add_argument('--subgraph_size', type=int, default=35)
    parser.add_argument('--n_hop', type=int, help='hop of neighbor nodes', default=5)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=6000)
    parser.add_argument('--lamda', type=float, help='a hyperparameter in loss function ', default=0.5)
    return parser

def main():

    torch.cuda.empty_cache()

    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        exit()
    print(args)

    # load data
    adj, x = load_dataset(args)
    edge = edge_list(adj)

    Gdata = Data(x=torch.Tensor(x), edge_index=edge)
    num_node = Gdata.num_nodes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # setting up the subgraph extractor
    ppr_path = ".\dataset\\" + args.dataset + '\subgraph'
    subgraph = Subgraph(Gdata.x, Gdata.edge_index, ppr_path, args.subgraph_size, args.n_hop)
    subgraph.build()

    model = Model(
        hidden_channels= args.hidden_size,
        encoder1=Encoder(Gdata.num_features, args.hidden_size),
        encoder2=Encoder(Gdata.num_features, args.hidden_size),
        pool=Pool(in_channels=args.hidden_size),
        scorer=Scorer(args.hidden_size)
    )

    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    def train():
        # model
        model.to(device)
        model.train()
        optim.zero_grad()
        sample_idx = list(range(num_node))
        batch, index = subgraph.search(sample_idx)

        z, summary, glo_out = model(batch.x.to(device), batch.edge_index.to(device), Gdata.x.to(device), Gdata.edge_index.to(device), batch.batch.to(device), index.to(device))

        loss = model.loss(z, summary, glo_out, args.lamda)
        loss.backward()
        optim.step()

        return loss.item()

    def test(model):
        model.to(device)
        model.eval()
        with torch.no_grad():
            sample_idx = list(range(num_node))
            batch, index = subgraph.search(sample_idx)

            z, summary, glo_out = model(batch.x.to(device), batch.edge_index.to(device), Gdata.x.to(device),
                                        Gdata.edge_index.to(device), batch.batch.to(device), index.to(device))
            # define a readout function
            # cat local and global node embedding
            all_embedding = torch.cat((z, glo_out), dim=1)
            out_embedding = all_embedding * (torch.softmax(all_embedding, dim=1))
            # attention readout layer
            # residucal layer
            get_all_node_emb = all_embedding + out_embedding
        return get_all_node_emb, summary

    print('Start training !!!')

    for _ in range(350):
        loss = train()

    # node embedding and graph embedding
    node_feature, graph_feature = test(model)
    idx, sam_node_feature, sam_graph_feature, labels = load_feature(node_feature, graph_feature, args)

    fold = KFold(n_splits=5, shuffle=True, random_state=1234)
    auc, aupr, acc, pre, recall, mcc, f1, spe, sen = [], [], [], [], [], [], [], [], []
    Tpr, Fpr, recall_au, pre_au = [], [], [], []
    for i, (train, test) in enumerate(fold.split(idx)):

        batch_data_train = data.TensorDataset(sam_node_feature[train], sam_graph_feature[train], labels[train])
        batch_data_val = data.TensorDataset(sam_node_feature[test], sam_graph_feature[test], labels[test])

        train_data_loader = data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle=True)
        val_data_loader = data.DataLoader(batch_data_val, batch_size=args.batch_size, shuffle=True)
        clf = TransClf(hidden_size=args.hidden_size, num_heads=args.num_heads)
        optim_clf = torch.optim.AdamW(clf.parameters(), lr=0.001, weight_decay=1e-5)
        Loss = nn.BCELoss()
        Train_loss = []
        for i in range(350):
            train_loss = []
            clf.to(device)
            clf.train()
            for _, item in enumerate(train_data_loader):
                node_feature = item[0]
                graph_feature = item[1]
                label = item[2]
                optim_clf.zero_grad()
                pred = clf(node_feature.to(device), graph_feature.to(device))
                loss = Loss(pred, (label).to(torch.float).to(device))
                loss.backward()
                optim_clf.step()
                train_loss.append(loss.item())
            # print(f'{i + 1}th, loss = {np.mean(train_loss):.4f}')

        clf.to(device)
        clf.eval()
        with torch.no_grad():
            clf_auc, clf_aupr, clf_acc, clf_pre, clf_recall, clf_mcc, clf_f1, clf_spe, clf_sen = [], [], [], [], [], [], [], [], []
            clf_fpr, clf_tpr, clf_rec, clf_pr = [], [], [], []
            for _, item in enumerate(val_data_loader):
                node_feature = item[0]
                graph_feature = item[1]
                label = item[2]
                prob = clf(node_feature.to(device), graph_feature.to(device))
                prob = prob.detach().cpu().numpy()
                clf_auc.append(roc_auc_score(label[:, 1], prob[:, 1]))
                clf_aupr.append(average_precision_score(label[:, 1], prob[:, 1]))
                pred = np.where(prob >= 0.5, 1, prob)
                pred = np.where(pred != 1, 0, pred)
                clf_acc.append(accuracy_score(label[:, 1], pred[:, 1]))
                clf_pre.append(precision_score(label[:, 1], pred[:, 1]))
                clf_recall.append(recall_score(label[:, 1], pred[:, 1]))
                clf_mcc.append(matthews_corrcoef(label[:, 1], pred[:, 1]))
                clf_f1.append(f1_score(label[:, 1], pred[:, 1]))
                tn, fp, fn, tp = confusion_matrix(label[:, 1], pred[:, 1]).ravel()
                clf_spe.append(tn / (tn + fp))
                clf_sen.append(tp / (tp + fn))
                fpr, tpr, _ = roc_curve(label[:, 1], prob[:, 1])
                pr, re, _ = precision_recall_curve(label[:, 1], prob[:, 1])
                clf_tpr.append(tpr)
                clf_fpr.append(fpr)
                clf_rec.append(re)
                clf_pr.append(pr)
                Tpr.append(tpr)
                Fpr.append(fpr)
                recall_au.append(re)
                pre_au.append(pr)

        # print(f'validation:auc:{clf_auc:.4f}, aupr:{clf_aupr:.4f}, acc:{clf_acc:.4f}, pre:{clf_pre:.4f}, recall:{clf_recall:.4f}, mcc:{clf_mcc:.4f}')
        auc.append(np.mean(clf_auc))
        aupr.append(np.mean(clf_aupr))
        acc.append(np.mean(clf_acc))
        pre.append(np.mean(clf_pre))
        recall.append(np.mean(clf_recall))
        mcc.append(np.mean(clf_mcc))
        f1.append(np.mean(clf_f1))
        spe.append(np.mean(clf_spe))
        sen.append(np.mean(clf_sen))
        print(
            f'Validation:auc:{np.mean(clf_auc):.4f}, aupr:{np.mean(clf_aupr):.4f}, acc:{np.mean(clf_acc):.4f}, pre:{np.mean(clf_pre):.4f}, recall:{np.mean(clf_recall):.4f}, mcc:{np.mean(clf_mcc):.4f}, '
            f'f1:{np.mean(clf_f1):.4f}, spe:{np.mean(clf_spe):.4f}, sen:{np.mean(clf_sen):.4f}')

    print(
        f' Final Validation:auc:{np.mean(auc):.4f}, aupr:{np.mean(aupr):.4f}, acc:{np.mean(acc):.4f}, pre:{np.mean(pre):.4f}, recall:{np.mean(recall):.4f}, mcc:{np.mean(mcc):.4f}, '
        f'f1:{np.mean(f1):.4f}, spe:{np.mean(spe):.4f}, sen:{np.mean(sen):.4f}')

    return np.mean(auc), np.mean(aupr), np.mean(acc), np.mean(pre), np.mean(recall), np.mean(
        mcc), np.mean(f1), np.mean(spe), np.mean(sen)

if __name__ == '__main__':
    Auc, Aupr, Acc, Pre, Recall, Mcc, F1, Spe, Sen = [], [], [], [], [], [], [], [], []
    for _ in range(10):
        auc, aupr, acc, pre, recall, mcc, f1, spe, sen = main()
        Auc.append(auc)
        Aupr.append(aupr)
        Acc.append(acc)
        Pre.append(pre)
        Recall.append(recall)
        Mcc.append(mcc)
        F1.append(f1)
        Spe.append(spe)
        Sen.append(sen)
    print(
        f"Final Result : Auc:{np.mean(Auc):.4f}, Auc_Std:{np.std(Auc):.4f}, Aupr:{np.mean(Aupr):.4f}, Aupr_Std:{np.std(Aupr):.4f}, "
        f"Acc:{np.mean(Acc):.4f}, Acc_Std:{np.std(Acc):.4f}, Pre:{np.mean(Pre):.4f}, Pre_Std:{np.std(Pre):.4f}, "
        f"Recall:{np.mean(Recall):.4f}, Recall_Std:{np.std(Recall):.4f}, Mcc:{np.mean(Mcc):.4f}, Mcc_Std:{np.std(Mcc):.4f}, "
        f"F1:{np.mean(F1):.4f}, F1_Std:{np.std(F1):.4f}, Spe:{np.mean(Spe):.4f}, Spe_Std:{np.std(Spe):.4f}, Sen:{np.mean(Sen):.4f}, Sen_Std:{np.std(Sen):.4f}")
