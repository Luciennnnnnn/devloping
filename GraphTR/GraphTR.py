# coding=utf-8
import os
import math
import heapq
import logging
import time
import numpy as np
import torch
import sys
import torch.nn as nn


def topk(X, _k, dimY):
    a = []
    for i in range(dimY[0]):
        for j in range(dimY[1]):
            for k in range(dimY[2]):
                a.append((X[i, j, k], (i, j, k)))
    mx = heapq.nlargest(_k, a, key=lambda s: abs(s[0]))
    return mx


class LHS:
    def __init__(self, A, B, W):
        self.A = A
        self.B = B
        self.W = W

    def H(self, c):
        c = torch.unsqueeze(c, 0)
        p = torch.squeeze(torch.mm(c, self.A))
        p = ((p + self.B) / self.W).floor()
        p = math.floor(p.sum().item() / 10)
        return p

    def Off(self, c):
        c = torch.unsqueeze(c, 0)
        delta = torch.squeeze(torch.mm(c, self.A)) + self.B
        delta = (delta.sum().item() / 10) % self.W
        return delta

    def dis(self, x, y):
        if x[0] == y[0]:
            return y[1] - x[1]
        else:
            return y[1] + self.W - x[1]

    def construct_graph(self, C, sz):
        # 没必要分bucket实现
        _arr = []
        for i in range(sz):
            p = self.H(C[i, :])  # 向量所处bucket编号
            off = self.Off(C[i, :])  # 所处位置距离初始处的偏移量
            _arr.append((p, off, i))

        _arr.sort(key=lambda s: (s[0], s[1]))  # 将所有向量进行排序，为后面建图做准备
        locations, q = [], []
        head, tail = 0, -1
        for x in _arr:
            while head <= tail and (q[head][0] < x[0] - 1 or self.dis(q[head], x) > self.W // 2):
                head += 1
            for i in range(head, tail + 1):
                locations.append((q[i][2], x[2]))
            q.append(x)
            tail += 1
        return locations


class Liner(torch.nn.Module):
    def __init__(self, I, J, K, R):
        super(Liner, self).__init__()
        self.A = nn.Parameter(torch.randn(I, R))  # nn.Parameter是特殊Variable
        self.B = nn.Parameter(torch.randn(J, R))
        self.C = nn.Parameter(torch.randn(K, R))

    def forward(self):
        _X = torch.einsum("ir,jr,kr->ijk", self.A, self.B, self.C)
        return _X


class GraphTR:
    def __init__(self, X, epsilon, R, W, max_epoch=30, maxiters=30):
        #R, W = 15, 2.5
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.R = R
        self.W = W
        self.epsilon = epsilon
        self.X = torch.from_numpy(X).cuda()
        self.R = R
        self.max_epoch = max_epoch
        self.maxiters = maxiters
        #a, b = [], []
        #D = [0 for _ in range(K)]
        self.criterion = torch.nn.MSELoss(reduction="sum")
        A = torch.randn([R, 10])
        B = torch.rand(10) * W
        A = A.to(self.device)
        B = B.to(self.device)
        self.lhs = LHS(A, B, W)
        self.learning_rate = 0
        self.WARMUP_STEPS = 800
        self.INIT_LR = 1e-4
        self.LRS = [5e-6, 7e-6, 9e-6, 1e-5, 3e-5, 5e-5, 7e-5, 9e-5, 1e-4, 1e-4, 1e-4, 1e-4, 2e-4,
               2e-4, 2e-4, 3e-4, 3e-4, 3e-4, 5e-4, 5e-4]

    def adjust_learning_rate(self, optimizer, train_steps):
        if self.WARMUP_STEPS and train_steps < self.WARMUP_STEPS:
            warmup_percent_done = train_steps / self.WARMUP_STEPS
            warmup_learning_rate = self.INIT_LR * warmup_percent_done  # gradual warmup_lr
            self.learning_rate = warmup_learning_rate
        else:
            # learning_rate = np.sin(learning_rate)  #预热学习率结束后,学习率呈sin衰减
            self.learning_rate = self.learning_rate ** 1.0001  # 预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def calc(self, V, F):
        loss = 0
        for f in F:
            loss += self.criterion(V[f[0], :], V[f[1], :])
        return loss * 2

    def run(self):
        device1 = torch.device("cpu")
        device2 = torch.device("cuda:0")  # 取消注释以在GPU上运行
        #torch.cuda.set_device(0)
        DIM = self.X.shape
        model = Liner(DIM[0], DIM[1], DIM[2], self.R).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        e = torch.zeros_like(self.X)
        e = e.cuda()
        print('asdasdasd:', self.device)
        for epoch in range(self.max_epoch):
            print(epoch)
            sys.stdout.flush()
            C = self.X - e
            # sub_problem-1
            start = time.time()
            for t in range(self.maxiters):
                # 前向传播：通过向模型传入x计算预测的y。
                self.adjust_learning_rate(optimizer, epoch * 100 + t + 1)
                start = time.time()
                _X = model()
                end = time.time()
                print('model():', end - start)
                total_loss = 0
                total_loss += self.criterion(_X, C) * 0.3

                start = time.time()
                F_a = self.lhs.construct_graph(model.A, DIM[0])
                F_b = self.lhs.construct_graph(model.B, DIM[1])
                F_c = self.lhs.construct_graph(model.C, DIM[2])
                end = time.time()
                print('construction:', end - start)

                start = time.time()
                total_loss += self.calc(model.A, F_a) * 0.001
                total_loss += self.calc(model.B, F_b) * 0.001
                total_loss += self.calc(model.C, F_c) * 0.001
                end = time.time()
                print('calc:', end-start)
                # loss2 *= 0.001
                # 清零梯度，反向传播，更新权重
                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                optimizer.step()
            end = time.time()
            print(end - start)
            sys.stdout.flush()
            # sub_problem-2
            start = time.time()
            _X = model()
            S = self.X - _X
            p = topk(S, self.epsilon, DIM)
            e = torch.zeros_like(self.X).to(self.device)
            for elem in p:
                s = elem[1]
                e[s[0], s[1], s[2]] = elem[0]
            end = time.time()
            print(end - start)
            sys.stdout.flush()
        return {'L': _X.detach().numpy(), 'S': e.detach().numpy()}