#coding:utf-8  将异常部分变为条件
import time
import os
import heapq
import copy
import json
import sys
import logging
from math import pi, pow, sqrt, log, log10
import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.random import random_kruskal
from tensorly.decomposition import parafac
from scipy.special import digamma, gamma
from scipy.stats import zscore
from imageio import imread, imsave
import scipy.signal
from numpy.linalg import norm, svd, inv, det, pinv
import matplotlib

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
logging.basicConfig(filename='logs/' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '.log', filemode="w", 
        format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

def clean(serie):
    output = serie[(np.isnan(serie) == False) & (np.isinf(serie) == False)]
    return output


def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), 'F')


def topk(X, _k, dimY):
    a = []
    for i in range(dimY[0]):
        for j in range(dimY[1]):
            for k in range(dimY[2]):
                a.append((X[i, j, k], (i, j, k)))
    mx = heapq.nlargest(_k, a, key=lambda s: abs(s[0]))
    return mx


def topk2(X, _k, dimY):
    a = []
    for i in range(dimY[0]):
        for j in range(dimY[1]):
            a.append((X[i, j], (i, j)))
    mx = heapq.nlargest(_k, a, key=lambda s: abs(s[0]))
    return mx


def DRMF(Y, epsilon, R, maxiters, init):
    S = np.zeros_like(Y)
    for it in range(maxiters):
        X = Y - S
        kruskal_tensor = parafac(X, rank=R, n_iter_max=15, init=init)
        S = Y - tl.kruskal_to_tensor(kruskal_tensor)
        p = topk(S, epsilon, S.shape)
        S = np.zeros_like(X)
        for elem in p:
            s = elem[1]
            S[s[0], s[1], s[2]] = elem[0]
    X = Y - S
    return [X, S, kruskal_tensor[0], kruskal_tensor[1], kruskal_tensor[2]]


def update(t, M, X, S, epsilon, maxiters, A_old, B_old, C_old, T_A1_old, T_A2_old, T_B1_old, T_B2_old, T):
    C = np.zeros_like(C_old)
    C[0:-1, :] = C_old[1:, :]
    C[-1, :] = np.dot(unfold(np.expand_dims(X[:, :, t+T], axis=2), 2), scipy.linalg.pinv(np.nan_to_num(khatri_rao([B_old, A_old]).T)))

    T_A1_com = T_A1_old - np.dot(unfold(np.expand_dims(X[:, :, t], axis=2), 0),
                                 khatri_rao([np.expand_dims(C_old[0, :], axis=0), B_old]))
    T_A1 = T_A1_com + np.dot(unfold(X[:, :, t+T], 0),
                             khatri_rao([np.expand_dims(C[-1, :], axis=0), B_old]))

    T_A2_com = T_A2_old - np.dot(np.expand_dims(C_old[0, :], axis=0).T,
                                 np.expand_dims(C_old[0, :], axis=0)) * np.dot(B_old.T, B_old)
    T_A2 = T_A2_com + np.dot(np.expand_dims(C[-1, :], axis=0).T,
                             np.expand_dims(C[-1, :], axis=0)) * np.dot(B_old.T, B_old)
    T_A2 = np.nan_to_num(T_A2)
    A = np.dot(T_A1, scipy.linalg.pinv(T_A2))

    T_B1_com = T_B1_old - np.dot(unfold(np.expand_dims(X[:, :, t], axis=2), 1),
                                 khatri_rao([np.expand_dims(C_old[0, :], axis=0), A_old]))
    T_B1 = T_B1_com + np.dot(unfold(X[:, :, t + T], 1),
                             khatri_rao([np.expand_dims(C[-1, :], axis=0), A_old]))

    T_B2_com = T_B2_old - np.dot(np.expand_dims(C_old[0, :], axis=0).T,
                                 np.expand_dims(C_old[0, :], axis=0)) * np.dot(A_old.T, A_old)
    T_B2 = T_B2_com + np.dot(np.expand_dims(C[-1, :], axis=0).T,
                             np.expand_dims(C[-1, :], axis=0)) * np.dot(A_old.T, A_old)
    T_B2 = np.nan_to_num(T_B2)
    B = np.dot(T_B1, scipy.linalg.pinv(T_B2))

    S[:, :, t + T] = np.squeeze(np.expand_dims(M[:, :, t + T], axis=2) -
                                tl.kruskal_to_tensor([A, B, np.expand_dims(C[-1, :], axis=0)], None))
    p = topk2(S[:, :, t + T], epsilon[t + T], S.shape)
    S[:, :, t + T] = np.zeros([S.shape[0], S.shape[1]])
    for elem in p:
        s = elem[1]
        S[s[0], s[1], t + T] = elem[0]

    for it in range(maxiters-1):
        X[:, :, t + T] = M[:, :, t + T] - S[:, :, t + T]
        #print('pinv(khatri_rao([B_old, A_old]).T):', scipy.linalg.pinv(khatri_rao([B_old, A_old]).T))
        C[-1, :] = np.dot(unfold(np.expand_dims(X[:, :, t+T], axis=2), 2), scipy.linalg.pinv(np.nan_to_num(khatri_rao([B_old, A_old]).T)))
        T_A1 = T_A1_com + np.dot(unfold(X[:, :, t + T], 0),
                                 khatri_rao([np.expand_dims(C[-1, :], axis=0), B_old]))

        T_A2 = T_A2_com + np.dot(np.expand_dims(C[-1, :], axis=0).T,
                             np.expand_dims(C[-1, :], axis=0)) * np.dot(B_old.T, B_old)

        tmp = np.dot(unfold(X[:, :, t + T], 0),
                                 khatri_rao([np.expand_dims(C[-1, :], axis=0), B_old]))

        tmp1 = khatri_rao([np.expand_dims(C[-1, :], axis=0), B_old])

        # if np.any(np.isnan(tmp1)):
        #     print("tmp1 is nan", it)
        #
        # if np.any(np.isinf(tmp1)):
        #     print("tmp1 is inf", it)
        #
        # if np.any(np.isnan(tmp)):
        #     print("tmp is nan", it)
        #
        # if np.any(np.isinf(tmp)):
        #     print("tmp is inf", it)
        #     print('B_old: ', B_old)
        #     print('A_old: ', A_old)
        #     print('khatri_rao([B_old, A_old]): ', khatri_rao([B_old, A_old]))
        #     print('pinv(khatri_rao([B_old, A_old]).T): ', scipy.linalg.pinv(khatri_rao([B_old, A_old]).T))
        #     print('unfold(np.expand_dims(X[:, :, t+T], axis=2), 2):', unfold(np.expand_dims(X[:, :, t+T], axis=2), 2))
        #     print('C[-1, :]: ', C[-1, :])
        #
        # if np.any(np.isnan(T_A1)):
        #     print("T_A1 is nan", it)
        #
        # if np.any(np.isinf(T_A1)):
        #     print("T_A1 is inf", it)
        #
        # if np.any(np.isnan(T_A2)):
        #     print("T_A2 is nan", it)
        #
        # if np.any(np.isinf(T_A2)):
        #     print("T_A2 is inf", it)

        T_A2 = np.nan_to_num(T_A2)
        A = np.dot(T_A1, scipy.linalg.pinv(T_A2))

        T_B1 = T_B1_com + np.dot(unfold(np.expand_dims(X[:, :, t + T], axis=2), 1),
                             khatri_rao([np.expand_dims(C[-1, :], axis=0), A_old]))
        T_B2 = T_B2_com + np.dot(np.expand_dims(C[-1, :], axis=0).T,
                             np.expand_dims(C[-1, :], axis=0)) * np.dot(A_old.T, A_old)

        T_B2 = np.nan_to_num(T_B2)
        B = np.dot(T_B1, scipy.linalg.pinv(T_B2))

        # if np.any(np.isnan(A)):
        #     print("A is nan", it)
        #
        # if np.any(np.isinf(A)):
        #     print("A is inf", it)
        #
        # if np.any(np.isnan(B)):
        #     print("B is nan", it)
        #
        # if np.any(np.isinf(B)):
        #     print("B is inf", it)

        S[:, :, t + T] = np.squeeze(np.expand_dims(M[:, :, t + T], axis=2) -
                                    tl.kruskal_to_tensor([A, B, np.expand_dims(C[-1, :], axis=0)], None))
        p = topk2(S[:, :, t + T], epsilon[t+T], S.shape)
        S[:, :, t+T] = np.zeros([S.shape[0], S.shape[1]])
        for elem in p:
            s = elem[1]
            S[s[0], s[1], t+T] = elem[0]

    X[:, :, t+T] = M[:, :, t+T] - S[:, :, t+T]
    return X, S, A, B, C, T_A1, T_A2, T_B1, T_B2


def T_online(Y, epsilon, R, W, maxiters, init):
    T = Y.shape[2]
    [X_old, S_old, A_old, B_old, C_old] = DRMF(Y[:, :, 0:W], np.sum(epsilon[0:W]), R, maxiters, init)
    Y = Y.astype(np.float64)
    X_old = X_old.astype(np.float128)
    S_old = S_old.astype(np.float128)
    A_old = A_old.astype(np.float128)
    B_old = B_old.astype(np.float128)
    C_old = C_old.astype(np.float128)
    X = np.zeros_like(Y, dtype=np.float128)
    X[:, :, 0:W] = X_old

    S = np.zeros_like(Y)
    S[:, :, 0:W] = S_old

    C = np.zeros_like(C_old)
    C[0:-1, :] = C_old[1:, :]
    C[-1, :] = np.dot(unfold(np.expand_dims(X[:, :, W], axis=2), 2), scipy.linalg.pinv(np.nan_to_num(khatri_rao([B_old, A_old]).T)))

    T_A1 = np.dot(unfold(X[:, :, 1:W], 0), khatri_rao([C[0:-1:, :], B_old])) + \
           np.dot(unfold(X[:, :, W], 0), khatri_rao([np.expand_dims(C[-1, :], axis=0), B_old]))
    T_A2 = np.dot(khatri_rao([C[0:-1, :], B_old]).T, khatri_rao([C[0:-1, :], B_old])) + \
           np.dot(khatri_rao([np.expand_dims(C[-1, :], axis=0), B_old]).T, khatri_rao([np.expand_dims(C[-1, :], axis=0), B_old]))
    T_A2 = np.nan_to_num(T_A2)
    A = np.dot(T_A1, scipy.linalg.pinv(T_A2))

    T_B1 = np.dot(unfold(X[:, :, 1:W], 1), khatri_rao([C[0:-1, :], A_old])) + \
           np.dot(unfold(X[:, :, W], 1), khatri_rao([np.expand_dims(C[-1, :], axis=0), A_old]))
    T_B2 = np.dot(khatri_rao([C[0:-1, :], A_old]).T, khatri_rao([C[0:-1, :], A_old])) + \
           np.dot(khatri_rao([np.expand_dims(C[-1, :], axis=0), A_old]).T, khatri_rao([np.expand_dims(C[-1, :], axis=0), A_old]))
    T_B2 = np.nan_to_num(T_B2)
    B = np.dot(T_B1, scipy.linalg.pinv(T_B2))

    for t in range(1, T-W):
        [X, S, A, B, C, T_A1, T_A2, T_B1, T_B2] = update(t, Y, X, S, epsilon, maxiters, A, B, C, T_A1, T_A2, T_B1, T_B2, W)

        #print('A', A.dtype)
        #print('B', B.dtype)
        #print('C', C.dtype)
        #print('X', X.dtype)
        #print('S', S.dtype)
        #print('T_A1', T_A1.dtype)
        #print('T_A2', T_A2.dtype)
        #print('T_B1', T_B1.dtype)
        #print('T_B2', T_B2.dtype)

    return X, S