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
from scipy.special import digamma, gamma
from scipy.stats import zscore
from imageio import imread, imsave
import scipy.signal
from numpy.linalg import norm, svd, inv, det
import matplotlib
matplotlib.use('Agg')
sys.path.append("../")

from utils import *

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
logging.basicConfig(filename='logs/' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '.log', filemode="w", 
        format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

def VITAD(Y, outliers_p, Omega, maxRank, maxiters, tol=1e-5, init='ml'):
    R = maxRank
    dimY = Y.shape
    N = Y.ndim
    T = dimY[N-1]
    outliers_count = np.sum(outliers_p, (0, 1))
    # Initialization
    a_tau0 = 1e-6
    b_tau0 = 1e-6

    tau = 1e4  # E[tau]
    dscale = 1
    X_C = np.zeros_like(Y)
    Z0 = [] # A^n的先验的期望
    ZSigma0 = [] # A^n的先验的方差

    for n in range(N-1):
        if init == 'rand':
            Z0.append(np.random.randn(dimY[n], R))  # E[A ^ (n)]
        elif init == 'ml':
            U, S, _ = np.linalg.svd(unfold(Y, n), full_matrices=False)
            S = np.diag(S)
            Z0.append(np.dot(U[:, :R], np.square(S[:R, :R])))
        ZSigma0.append(np.zeros([R, R, dimY[n]]))  # covariance of A ^ (n)
        for i in range(dimY[n]):
            ZSigma0[n][:, :, i] = np.eye(R)

    # U, S, _ = np.linalg.svd(unfold(Y, 2), full_matrices=False)
    # S = np.diag(S)
    # z_tmp = np.dot(U[:, :R], np.square(S[:R, :R]))

    Z = [] # A^n的后验的期望
    ZSigma = [] # A^n的后验的方差
    for n in range(N - 1):
        Z.append(Z0[n].copy())  # E[A ^ (n)]
        ZSigma.append(ZSigma0[n].copy())

    # --------- E(aa^T) = cov(a,a) + E(a)E(a^T)----------------
    EZZT = []
    for n in range(N-1):
        # EZZT.append(np.reshape(ZSigma[n], [R*R, dimY[n]], 'F').T)#向量化的B^(n)
        EZZT.append((np.reshape(ZSigma[n], [R * R, dimY[n]], 'F') + khatri_rao([Z[n].T, Z[n].T])).T) # 向量化的B^(n)
    #         EZZT{n} = (reshape(ZSigma{n}, [R*R, dimY(n)]))' + khatrirao_fast(Z{n}',Z{n}')'\

    Z0_t = np.random.randn(T, R)
    Z_t = np.random.randn(T, R)
    # elif init == 'ml':
    #     Z0_t = np.expand_dims(z_tmp[t, :], axis=0)
    #     Z_t = np.expand_dims(z_tmp[t, :], axis=0)

    ZSigma0_t = np.expand_dims(np.eye(R), 2)
    ZSigma0_t = ZSigma0_t.repeat(T, axis=2)
    ZSigma_t = np.zeros_like(ZSigma0_t)

    sigma_E0 = np.ones([dimY[0], dimY[1], T]) # \zeta, prior precision of \mathcal{E}
    E0 = np.zeros([dimY[0], dimY[1], T]) # actually not need
    # E0 = np.random.randn(dimY[0], dimY[1], 1)

    sigma_E = np.ones_like(sigma_E0) # \overline{\zeta}, posterior precision of \mathcal{E}
    E = np.zeros_like(E0) # \overline{\mathcal{E}}, posterior mean of \mathcal{E}
    
    RSE = []
    TPRS = []
    FPRS = []
    count = 0
    false_locations = []
    for t in range(T):
        if t % 800 == 0:
            logging.debug(t)
        #  Model learning
        LB = []
        O = np.expand_dims(Omega[:, :, t], axis=2)
        nObs = np.sum(O)
        EZZT_t = np.reshape(ZSigma0_t[:, :, t], [R * R, 1], 'F').T

        C = np.expand_dims(Y[:, :, t], 2)
        a_tauN = 1e-6
        b_tauN = 1e-6
        for it in range(maxiters):
            # Update factor matrices
            for n in range(N-1):
                # compute E(Z_{\n}^{T} Z_{\n})
                ENZZT = np.reshape(np.dot(khatri_rao([EZZT[0], EZZT[1], EZZT_t], skip_matrix=n, reverse=bool).T,
                                      unfold(O, n).T), [R, R, dimY[n]], 'F')
                # compute E(Z_{\n})
                FslashY = np.dot(khatri_rao([Z[0], Z[1], np.expand_dims(Z_t[t, :], axis=0)], skip_matrix=n, reverse=bool).T,
                                 unfold((C-np.expand_dims(E[:, :, t], axis=2)) * O, n).T)

                for i in range(dimY[n]):
                    #logging.debug('n:%d, i:%d Z'%(n, i))
                    #logging.debug('before ZSigma:' + str(ZSigma[n][:, :, i]))
                    ZSigma[n][:, :, i] = inv(tau * ENZZT[:, :, i] + inv(ZSigma0[n][:, :, i]))
                    #logging.debug('after ZSigma:' + str(ZSigma[n][:, :, i]))

                    #logging.debug('before Z:' + str(Z[n][i, :]))
                    Z[n][i, :] = np.squeeze((np.dot(ZSigma[n][:, :, i], (inv(ZSigma0[n][:, :, i]).dot(np.expand_dims(Z0[n][i, :], 1))
                                                    + tau*np.expand_dims(FslashY[:, i], 1)))).T)
                                                    
                    #logging.debug('after Z:' + str(Z[n][i, :]))

                EZZT[n] = (np.reshape(ZSigma[n], [R * R, dimY[n]], 'F') + khatri_rao([Z[n].T, Z[n].T])).T

            ENZZT = np.reshape(np.dot(khatri_rao([EZZT[0], EZZT[1], EZZT_t], skip_matrix=2, reverse=bool).T, unfold(O, 2).T),
                               [R, R, 1], 'F')
            # compute E(Z_{\n})
            FslashY = np.dot(khatri_rao([Z[0], Z[1], Z_t[t, :]], skip_matrix=2, reverse=bool).T,
                             unfold((C-E[:, :, t]) * O, 2).T)

            ZSigma_t[:, :, t] = inv(tau * ENZZT[:, :, 0] + inv(ZSigma0_t[:, :, t]))

            Z_t[t, :] = (np.dot(ZSigma_t[:, :, t], (inv(ZSigma0_t[:, :, t]).dot(np.reshape(Z0_t[t, :], [R, 1])) +
                                                                 tau*np.expand_dims(FslashY[:, 0], 1)))).T

            EZZT_t = (np.reshape(np.reshape(Z_t[t, :], [R, 1]).dot(np.expand_dims(Z_t[t, :], axis=0)), [R*R, 1], 'F') + \
                             np.reshape(ZSigma_t[:, :, t], [R*R, 1], 'F')).T

            # Update latent tensor X
            X = np.squeeze(tl.cp_to_tensor((None, [Z[0], Z[1], np.expand_dims(Z_t[t, :], axis=0)])))   #

            #  The most time and space consuming part
            EX2 = np.dot(np.dot(tensor_to_vec(O).T, khatri_rao([EZZT[0], EZZT[1],
                                EZZT_t], reverse=bool)), np.ones([R * R, 1]))

            EE2 = np.sum(np.square(E[:, :, t]) * np.squeeze(O) + np.reciprocal(sigma_E[:, :, t]) * np.squeeze(O))
            err = np.dot(tensor_to_vec(C*O).T, tensor_to_vec(C*O)) \
                    - 2 * tensor_to_vec(C*O).T.dot(tensor_to_vec(X*np.squeeze(O))) \
                    - 2*np.dot(tensor_to_vec(C*O).T, tensor_to_vec(O)*tensor_to_vec(E[:, :, t])) \
                    + 2*np.dot(tensor_to_vec(X*np.squeeze(O)).T, tensor_to_vec(E[:, :, t])) + EX2 + EE2

            # update noise tau
            a_tauN = (a_tau0 + 0.5 * nObs)  # a_MnObs
            b_tauN = (b_tau0 + 0.5 * err)  # b_M

            tau = a_tauN / b_tauN # posterior mean of tau 

            # Update sparse matrix E
            sigma_E[:, :, t] = tau + sigma_E0[:, :, t]
            E[:, :, t] = (tau * np.squeeze(C-np.expand_dims(X, axis=2)) + sigma_E0[:, :, t]*E0[:, :, t]) / sigma_E[:, :, t]

            # calculate lowerbound
            # all the const part in bottom is actually not needed, can be removed for more fast calculattion.

            # E_q[lnp(\mathcal{Y} | \mathcal{E}, \tau^{-1})]
            temp1 = 0.5 * nObs * (digamma(a_tauN) - safelog(b_tauN)) - 0.5 * tau * err \
                    - 0.5 * nObs * safelog(2 * pi) # const

            # E_q[lnp(A)]
            temp2 = 0 
            for n in range(N-1):
                for i in range(dimY[n]):
                  temp2 += - 0.5 * np.trace(inv(ZSigma0[n][:, :, i]).dot(ZSigma[n][:, :, i])) \
                        - 0.5 * np.dot(np.expand_dims(Z[n][i, :], 0), inv(ZSigma0[n][:, :, i])).dot(np.expand_dims(Z[n][i, :], 1)) \
                        + np.dot(np.expand_dims(Z[n][i, :], 0), inv(ZSigma0[n][:, :, i])).dot(np.expand_dims(Z0[n][i, :], 1)) \
                         -0.5 * np.dot(np.expand_dims(Z0[n][i, :], 0), inv(ZSigma0[n][:, :, i])).dot(np.expand_dims(Z0[n][i, :], 1)) \
                         -0.5 * R * safelog(2 * pi) - 0.5 * safelog(det(ZSigma0[n][:, :, i])) # last two line const

            temp2 += - 0.5 * np.trace(inv(ZSigma0_t[:, :, t]).dot(ZSigma_t[:, :, t])) \
                    - 0.5 * np.squeeze(np.dot(np.expand_dims(Z_t[t, :], axis=0), inv(ZSigma0_t[:, :, t])).dot(np.reshape(Z_t[t, :], [R, 1]))) \
                    + np.squeeze(np.dot(np.expand_dims(Z_t[t, :], axis=0), inv(ZSigma0_t[:, :, t])).dot(np.reshape(Z0_t[t, :], [R, 1]))) \
                    -0.5 * np.dot(np.expand_dims(Z0_t[t, :], 0), inv(ZSigma0_t[:, :, t])).dot(np.expand_dims(Z0_t[t, :], 1)) \
                    -0.5 * R * safelog(2 * pi) - 0.5 * safelog(det(ZSigma0_t[:, :, t])) # last two line const

            # E_q[lnp(\tau)]
            temp3 = (a_tau0 - 1) * (digamma(a_tauN) - safelog(b_tauN)) - b_tau0 * tau \
                    -safelog(gamma(a_tau0)) + a_tau0 * safelog(b_tau0) # const

            # E_q[lnq(A)]
            temp4 = 0
            for n in range(N-1):
                for i in range(dimY[n]):
                    temp4 += 0.5 * safelog(det(ZSigma[n][:, :, i])) \
                        + 0.5 * R * (1 + safelog(2 * pi)) # consts

            temp4 += 0.5 * safelog(det(ZSigma_t[:, :, t])) \
                    + 0.5 * R * (1 + safelog(2 * pi)) # const

            # -E_q[lnq(\tau)]
            temp5 = safelog(gamma(a_tauN)) - (a_tauN - 1) * digamma(a_tauN) - safelog(b_tauN) + a_tauN

            #E_q[lnp(\mathcal{E})]
            temp6 = -0.5 * np.sum(sigma_E0[:, :, t] * (np.square(E[:, :, t]) + np.reciprocal(sigma_E[:, :, t])
                                - 2*E[:, :, t]*E0[:, :, t])) \
                    -0.5 * nObs * safelog(2 * pi) + 0.5 * np.sum(safelog(sigma_E0)) \
                    -0.5 * np.sum(sigma_E0[:, :, t] * np.square(E0[:, :, t])) # last two line const

            # -E_q[lnq(\mathcal{E})]
            temp7 = - 0.5 * np.sum(safelog(sigma_E[:, :, t])) \
                    + 0.5 * nObs * (safelog(2 * pi) + 1) # const

            LB.append(temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7)

            # Display progress
            if it > 2:
                LBRelChan = abs(LB[it] - 2*LB[it-1] + LB[it-2])/-LB[1]
            else:
                LBRelChan = 0
            #
            if t % 800 == 0:
                logging.debug('Iter. %d: RelChan = %g LB = %g' % (it, LBRelChan, LB[it]))
                
            
            # Convergence check
            if it > 5 and (abs(LBRelChan) < tol):
                # print('======= Converged===========')
                if t % 800 == 0:
                    logging.debug('======= Converged (%d, %d)===========' % (t, it))
                break
        # if t % 800 == 0:
        #     logging.debug('t %d, Iter. %d' %(t, it))
            
        #     logging.debug('Z0[0]:' + str(Z0[0]))
        #     logging.debug('Z0[1]:' + str(Z0[1]))
        #     logging.debug('ZSigma0[0]:' + str(ZSigma0[0]))
        #     logging.debug('ZSigma0[1]:' + str(ZSigma0[1]))
        #     logging.debug('Z[0]:' + str(Z[0]))
        #     logging.debug('Z[1]:' + str(Z[1]))
        #     logging.debug('ZSigma[0]:' + str(ZSigma[0]))
        #     logging.debug('ZSigma[1]:' + str(ZSigma[1]))


        #     logging.debug('a_tau0: %f' %(a_tau0))
        #     logging.debug('b_tau0: %f' %(b_tau0))
        #     logging.debug('a_tauN: %f' %(a_tauN))
        #     logging.debug('b_tauN: %f' %(b_tauN))
            
        #     logging.debug('E0:' + str(E0[:, :, t]))
        #     logging.debug('sigma_E0:' + str(sigma_E0[:, :, t]))
        #     logging.debug('E:' + str(E[:, :, t]))
        #     logging.debug('sigma_E:' + str(sigma_E[:, :, t]))

        [TPR, FPR] = check(E[:, :, t], outliers_p[:, :, t], outliers_count[t], dimY)
        #false_locations.append(locations)
        count += TPR
        #X = np.expand_dims(X, 2)
        if t % 800 == 0:
            X = np.squeeze(tl.cp_to_tensor((None, [Z[0], Z[1], np.expand_dims(Z_t[t, :], axis=0)])))
            logging.debug('t: %d, RSE: %f' % (t, norm(np.squeeze(C)-X-E[:, :, t]) / norm(C)))
        TPRS.append(TPR)
        FPRS.append(FPR)
        Z0 = copy.deepcopy(Z)
        ZSigma0 = copy.deepcopy(ZSigma)
        X_C[:, :, t] = X
        # E0 = copy.deepcopy(E)
        # sigma_E0 = copy.deepcopy(sigma_E0)
        a_tau0 = a_tauN
        b_tau0 = b_tauN

    # Prepare the results
    #Z.append(Z_ST)

    model = {}
    # Output
    model['X'] = X_C
    model['X2'] = tl.cp_to_tensor((None, [Z[0], Z[1], Z_t]))
    model['E'] = E
    #model['RSE'] = RSE
    model['TPRS'] = TPRS
    model['FPRS'] = FPRS
    model['precision'] = count / T
    model['FPR'] = np.sum(FPRS) / T
    #model['ZSigma'] = ZSigma
    #model['false_locations'] = false_locations
    return model