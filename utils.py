import copy
import json
import os
import heapq
import logging
from math import pi, pow, sqrt, log, log10
import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.random import random_kruskal
from scipy.special import digamma, gamma
from scipy.stats import zscore, exponpow
import scipy.signal
from numpy.linalg import norm, svd, inv, det
from scipy.stats import levy_stable


def safelog(x):
    if isinstance(x, np.ndarray):
        x[np.where(x < 1e-300)] = 1e-200
        x[np.where(x > 1e300)] = 1e300
        return np.log(x)
    else:
        if x < 1e-300:
            x = 1e-200
        if x > 1e300:
            x = 1e300
        return log(x)


def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), 'F')


def tensor_to_vec(tensor):
    return np.reshape(tensor, (-1, 1), 'F')


def hardmard(tensors):
    ans = np.ones_like(tensors[0])
    for i in range(len(tensors)):
        ans = ans * tensors[i]
    return ans


def choice(DIM, p):
    indices = np.random.choice(np.prod(DIM), int(round(np.prod(DIM) * p)), replace=False)
    X, Y, Z = [], [], []
    for i in range(indices.size):
        Z.append(indices[i] // (DIM[0] * DIM[1]))
        indices[i] %= DIM[0] * DIM[1]
        Y.append(indices[i] // DIM[0])
        X.append(indices[i] % DIM[0])
    return X, Y, Z


def generator(dataset_name, parameters):
    a = np.load(os.path.join(os.path.join('../../data', dataset_name), r'normlized_tensor.npy'))
    b = np.zeros_like(a, dtype=bool)
    omega = np.ones_like(a)
    DIM = a.shape
    locations = list(range(np.prod(DIM)))

    sigma = parameters['sigma']
    mu = parameters['mu']
    fraction = parameters['fraction']
    missing_ratio = parameters['missing_ratio']
    # missing..
    if missing_ratio != 0:
        missing_locations = np.random.choice(locations, int(len(locations) * missing_ratio), replace=False)
        for x in missing_locations:
            k = x // (DIM[0] * DIM[1])
            x %= (DIM[0] * DIM[1])
            i = x // DIM[1]
            j = x % DIM[1]
            omega[i, j, k] = 0

    noises = np.zeros_like(a)
    #Add noise
    if parameters['noise_scheme'] == "Gaussian":
        gaussian_noise = np.random.normal(0, 0.01, (DIM[0], DIM[1], DIM[2])) # 100%
        noises += gaussian_noise
    elif parameters['noise_scheme'] == "outlier":
        outlier_noise = np.random.uniform(-0.05, 0.05, (DIM[0], DIM[1], DIM[2]))
        
        sampled_locations = np.random.choice(locations, int(len(locations) * 0.1), replace=False)
        # print(len(sampled_locations))
        for x in sampled_locations:
            k = x // (DIM[0] * DIM[1])
            x %= (DIM[0] * DIM[1])
            i = x // DIM[1]
            j = x % DIM[1]
            noises[i, j, k] += outlier_noise[i, j, k]
    elif parameters['noise_scheme'] == "mixture":
        outlier_noise = np.random.uniform(-0.05, 0.05, (DIM[0], DIM[1], DIM[2])) # 10%
        gaussian_noise_1 = np.random.normal(0, 0.1, (DIM[0], DIM[1], DIM[2])) # 30%
        gaussian_noise_2 = np.random.normal(0, 0.2, (DIM[0], DIM[1], DIM[2])) # 20%
        # rvs(b, loc=0, scale=1, size=1, random_state=None)
        expoential_power_noise = exponpow.rvs(0.5, loc=0, scale=0.1, size=(DIM[0], DIM[1], DIM[2]), random_state=None) # 20%

        sampled_locations = np.random.choice(locations, int(len(locations) * 0.8), replace=False)
        # print(len(sampled_locations))
        for id in range(0, len(sampled_locations)):
            x = sampled_locations[id]
            k = x // (DIM[0] * DIM[1])
            x %= (DIM[0] * DIM[1])
            i = x // DIM[1]
            j = x % DIM[1]
            if id < int(len(locations)) * 0.1:
                noises[i, j, k] += outlier_noise[i, j, k]
            elif id < int(len(locations)) * 0.4:
                noises[i, j, k] += gaussian_noise_1[i, j, k]
            elif id < int(len(locations)) * 0.6:
                noises[i, j, k] += gaussian_noise_2[i, j, k]
            else:
                noises[i, j, k] += expoential_power_noise[i, j, k]
            

    # if SNR != None:
    #     sigma2 = np.var(tensor_to_vec(a))*(1 / (10 ** (SNR / 10)))
    #     GN = np.sqrt(sigma2) * np.random.randn(DIM[0], DIM[1], DIM[2])
    #     a = a + GN

    #Add outliers
    
    inject_outliers = np.zeros_like(a)

    if parameters['outliers_scheme'] == "Gaussian":
        if fraction != 0:
            outliers = np.random.randn(DIM[0], DIM[1], DIM[2]) * sqrt(sigma) + mu
            sampled_locations = np.random.choice(locations, int(len(locations) * fraction), replace=False)
            for x in sampled_locations:
                k = x // (DIM[0] * DIM[1])
                x %= (DIM[0] * DIM[1])
                i = x // DIM[1]
                j = x % DIM[1]
                b[i, j, k] = 1
                inject_outliers[i, j, k] += outliers[i, j, k]

    elif parameters['outliers_scheme'] == "Exponential":
        if fraction != 0:
            outliers = np.random.exponential(scale=0.5, size=(DIM[0], DIM[1], DIM[2]))
            sampled_locations = np.random.choice(locations, int(len(locations) * fraction), replace=False)
            for x in sampled_locations:
                k = x // (DIM[0] * DIM[1])
                x %= (DIM[0] * DIM[1])
                i = x // DIM[1]
                j = x % DIM[1]
                b[i, j, k] = 1
                inject_outliers[i, j, k] += outliers[i, j, k]
    elif parameters['outliers_scheme'] == "structural":
        outliers = np.random.uniform(0, 5, (DIM[0], DIM[1], DIM[2]))
        cur_locations = list(range(DIM[0], DIM[1]))
        for t in range(DIM[2]):
            sampled_locations = np.random.choice(cur_locations, int(len(cur_locations) * 0.01), replace=False)
            for x in sampled_locations:
                i = x // DIM[1]
                j = x % DIM[1]
                b[i, j, t] = 1
                inject_outliers[i, j, t] += outliers[i, j, t]

    elif parameters['outliers_scheme'] == "random":
        outliers = np.random.uniform(0, 5, (DIM[0], DIM[1], DIM[2]))
        sampled_locations = np.random.choice(locations, int(len(locations) * 0.01), replace=False)
        for x in sampled_locations:
            k = x // (DIM[0] * DIM[1])
            x %= (DIM[0] * DIM[1])
            i = x // DIM[1]
            j = x % DIM[1]
            b[i, j, k] = 1
            inject_outliers[i, j, k] += outliers[i, j, k]
    elif parameters['outliers_scheme'] == "levy_stable":
        outliers = levy_stable.rvs(sigma, mu, size=(DIM[0], DIM[1], DIM[2]))

    return a, inject_outliers, b, noises, omega


def generator2(dataset_name, fraction, mu, sigma, SNR=None, distribution="Gaussian"):
    #print(os.getcwd())
    a = np.load(os.path.join(os.path.join('../../data', dataset_name), r'normlized_tensor.npy'))
    b = np.zeros_like(a, dtype=bool)
    DIM = a.shape
    sigmas = [0.01, 0.05, 0.1, 0.5, 1]
    #Add noise
    if SNR != None:
        sigma2 = np.var(tensor_to_vec(a)) * (1 / (10 ** (SNR / 10)))
        GN = np.sqrt(sigma2) * np.random.randn(DIM[0], DIM[1], DIM[2])
        a = a + GN

    for it in range(5):
        #Add outliers
        if distribution == "Gaussian":
            outliers = np.random.randn(DIM[0], DIM[1], DIM[2]) * sqrt(sigmas[it]) + mu
        elif distribution == "levy_stable":
            outliers = levy_stable.rvs(sigma, mu, size=(DIM[0], DIM[1], DIM[2]))

        locations = list(range(np.prod(DIM)))
        if fraction != 0:
            sampled_locations = np.random.choice(locations, int(len(locations) * fraction), replace=False)
            # print(len(sampled_locations))
        for x in sampled_locations:
            k = x // (DIM[0] * DIM[1])
            x %= (DIM[0] * DIM[1])
            i = x // DIM[1]
            j = x % DIM[1]
            b[i, j, k] = 1
            a[i, j, k] += outliers[i, j, k]
    return a, b


def topk(X, _k, dimY):
    a = []
    for i in range(dimY[0]):
        for j in range(dimY[1]):
            a.append((X[i, j], (i, j)))
    mx = heapq.nlargest(_k, a, key=lambda s: abs(s[0]))
    return mx


def check(e, outliers_p, epsilon, dimY):
    TP = 0
    FP = 0
    p = topk(e, epsilon, dimY)
    #false_locations = []
    for elem in p:
        s = elem[1]
        if outliers_p[s[0], s[1]]:
            TP += 1
        else:
            FP += 1
            #false_locations.append((s[0], s[1]))
    if epsilon == 0:
        TPR = 1
    else:
        TPR = TP / epsilon
    if epsilon == dimY[0] * dimY[1]:
        FPR = 0
    else:
        FPR = FP / (dimY[0] * dimY[1] - epsilon)
    return TPR, FPR#, false_locations