from copy import deepcopy

import numpy as np
import pandas as pd


def dominate(sim_matrix, k = 20):
    """
    find the K nearest neighbors for each row in a similarity matrix
    :param sim_matrix: Input similarity matrix
    :param k: Number of K-nearest-neighbors ( default 20 )
    """
    def zero(sim):
        s = np.argsort(sim)
        sim[s[:len(sim) - k]] = 0
        return sim

    def normalize(X):
        # 归一化
        for i in range(X.shape[0]):
            rowsum = np.sum(X[i, :])
            for j in range(X.shape[1]):
                X[i, j] = X[i, j] / rowsum

        return X

    A = np.zeros(sim_matrix.shape)
    for i in range(len(sim_matrix)):
        A[i] = zero(deepcopy(sim_matrix[i]))
    return normalize(A)

def SNF(sim, k = 20, t = 20):
    """
    Perform similarity network fusion on a set of similarity matrices
    :param list_sim: list of similarity matrices
    :param k: number of k-nearest-neighbors ( default 20 )
    :param t: number of fusion iterations ( default 20 )
    :return: integrated similarity matrices
    """
    def normalize(X):
        # delete the elements at diag-line
        row_sums_less_diag = np.sum(X, axis=1, keepdims=True) - np.expand_dims(np.diagonal(X), axis=1)
        # if row sum = 0, set to 1 to avoid dividing by zero
        row_sums_less_diag[row_sums_less_diag == 0] = 1.0
        X = X / (2 * row_sums_less_diag)
        np.fill_diagonal(X, 0.5)
        return X

    LS = len(sim)
    newS = np.empty((LS, ) + sim[0].shape, dtype=np.float64)
    nextS = np.empty((LS, ) + sim[0].shape, dtype=np.float64)

    # convert arrays in sim_list to np.float64 data type
    sim = [arr.astype(np.float64) for arr in sim]

    # normalize similarity matrices
    for i in range(LS):
        sim[i] = normalize(sim[i])
        sim[i] = (sim[i] + sim[i].T) / 2.0

    # calculate the local similarity array using knn
    for i in range(LS):
        newS[i] = dominate(sim[i], k)

    # perform diffusion for t iterations
    for i in range(t):
        for j in range(LS):
            sumSJ = np.zeros(sim[j].shape, dtype=np.float64)
            for k in range(LS):
                if k != j:
                    sumSJ += sim[k]
            nextS[j] = newS[j].dot(sumSJ / (LS - 1)).dot(newS[j].T)
        # normalize each new network
        for j in range(LS):
            sim[j] = normalize(nextS[j])
            sim[j] = (sim[j] + sim[j].T) / 2.0

    # construct combined similarity matrix by summing diffused matrices
    S = np.zeros(sim[0].shape, dtype=np.float64)
    for i in range(LS):
        S += sim[i]
    S = S / float(LS)
    S = normalize(S)
    S = (S + S.T) / 2.0
    return S


