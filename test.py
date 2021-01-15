import matplotlib
import numpy as np
import os
import json
from matplotlib import pyplot as plt
matplotlib.use('Agg')


def plot_R(dataset_name):
    x = np.arange(5, 38, 3)
    pre = os.path.join(os.path.join('results', dataset_name))

    with open(os.path.join(pre, 'R_TPRS_mean.json'), 'r') as FR:
        y = json.loads(FR.read())
    with open(os.path.join(pre, 'R_TPRS_stderr.json'), 'r') as FR:
        dy = np.asarray(json.loads(FR.read()))
    plt.errorbar(x, y, yerr=dy, fmt='o--r', ecolor='r', elinewidth=2, capsize=4, label='asdas')
    plt.xlabel('R')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('R_TPRS.jpg')
    plt.clf()
    with open(os.path.join(pre, 'R_FPRS_mean.json'), 'r') as FR:
        y = json.loads(FR.read())
    with open(os.path.join(pre, 'R_FPRS_stderr.json'), 'r') as FR:
        dy = json.loads(FR.read())
    plt.errorbar(x, y, yerr=dy, fmt='o-r', ecolor='r', elinewidth=2, capsize=4)
    plt.savefig('R_FPRS.jpg')


def plot_mu(dataset_name):
    x = np.arange(1, 11, 1)
    pre = os.path.join(os.path.join('results', dataset_name))
    with open(os.path.join(pre, 'mu_TPRS_mean.json'), 'r') as FR:
        y = json.loads(FR.read())
    with open(os.path.join(pre, 'mu_TPRS_stderr.json'), 'r') as FR:
        dy = np.asarray(json.loads(FR.read()))

    plt.errorbar(x, y, yerr=dy, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)

    with open(os.path.join(pre, 'mu_FPRS_mean.json'), 'r') as FR:
        y = json.loads(FR.read())
    with open(os.path.join(pre, 'mu_FPRS_stderr.json'), 'r') as FR:
        dy = json.loads(FR.read())
    plt.errorbar(x, y, yerr=dy, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)


def plot_sigma(dataset_name):
    x = np.arange(1, 11, 1)
    pre = os.path.join(os.path.join('results', dataset_name))
    with open(os.path.join(pre, 'sigma_TPRS_mean.json'), 'r') as FR:
        y = json.loads(FR.read())
    with open(os.path.join(pre, 'sigma_TPRS_stderr.json'), 'r') as FR:
        dy = np.asarray(json.loads(FR.read()))

    plt.errorbar(x, y, yerr=dy, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)

    with open(os.path.join(pre, 'sigma_FPRS_mean.json'), 'r') as FR:
        y = json.loads(FR.read())
    with open(os.path.join(pre, 'sigma_FPRS_stderr.json'), 'r') as FR:
        dy = json.loads(FR.read())
    plt.errorbar(x, y, yerr=dy, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)


def plot_ratio(dataset_name):
    x = np.arange(1, 11, 1)
    pre = os.path.join(os.path.join('results', dataset_name))
    with open(os.path.join(pre, 'ratio_TPRS_mean.json'), 'r') as FR:
        y = json.loads(FR.read())
    with open(os.path.join(pre, 'ratio_TPRS_stderr.json'), 'r') as FR:
        dy = np.asarray(json.loads(FR.read()))

    plt.errorbar(x, y, yerr=dy, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)

    with open(os.path.join(pre, 'ratio_FPRS_mean.json'), 'r') as FR:
        y = json.loads(FR.read())

    with open(os.path.join(pre, 'ratio_FPRS_stderr.json'), 'r') as FR:
        dy = json.loads(FR.read())
    plt.errorbar(x, y, yerr=dy, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)


def plot_time(dataset_name, ed):
    x = np.arange(0, ed)
    pre = os.path.join(os.path.join('results', dataset_name))
    with open(os.path.join(pre, 'TPRS_mean.json'), 'r') as FR:
        y = json.loads(FR.read())
    with open(os.path.join(pre, 'TPRS_stderr.json'), 'r') as FR:
        dy = np.asarray(json.loads(FR.read()))

    plt.errorbar(x, y, yerr=dy, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)

    with open(os.path.join(pre, 'FPRS_mean.json'), 'r') as FR:
        y = json.loads(FR.read())

    with open(os.path.join(pre, 'FPRS_stderr.json'), 'r') as FR:
        dy = json.loads(FR.read())
    plt.errorbar(x, y, yerr=dy, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)


if __name__ == '__main__':
    x = [0.01, 0.05, 0.1, 0.5, 1]
    y = range(1, 6)
    y2 = range(2, 7)
    y3 = range(3, 8)
    y4 = range(4, 9)
    plt.plot(range(len(x)), y, color='#3d5d46', marker='o', linestyle='-')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':')
    plt.xticks(range(len(x)), x)
    plt.savefig('test.jpg')