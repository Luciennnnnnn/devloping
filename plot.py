import matplotlib
import numpy as np
import os
import json
from matplotlib import pyplot as plt
matplotlib.use('Agg')


def plot_R(dataset_name):
    x = np.arange(2, 51, 2)
    pre = os.path.join(os.path.join('results', dataset_name))

    with open(os.path.join(pre, os.path.join('proposed', 'R_TPRS.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(range(len(x)), y)
    plt.xticks(range(len(x)), x)
    plt.xlabel('R')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('R_TPRS.png')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'R_FPRS.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(range(len(x)), y)
    plt.xticks(range(len(x)), x)
    plt.xlabel('R')
    plt.ylabel('FPR')
    plt.legend()
    plt.savefig(os.path.join(pre, 'R_FPRS.png'))
    plt.clf()


def plot_mu(dataset_name):
    x = [0.01, 0.05, 0.1, 0.5, 1]
    pre = os.path.join(os.path.join('results', dataset_name))

    with open(os.path.join(pre, os.path.join('proposed', 'mu_TPRS.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(range(len(x)), y)
    plt.xticks(range(len(x)), x)
    plt.legend()
    plt.savefig('mu_TPRS.png')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'mu_FPRS.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(range(len(x)), y)
    plt.xticks(range(len(x)), x)
    plt.legend()
    plt.savefig(os.path.join(pre, 'mu_FPRS.png'))
    plt.clf()


def plot_sigma(dataset_name):
    x = [0.01, 0.05, 0.1, 0.5, 1]
    pre = os.path.join(os.path.join('results', dataset_name))

    with open(os.path.join(pre, os.path.join('proposed', 'sigma_TPRS.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(range(len(x)), y)
    plt.xticks(range(len(x)), x)
    plt.legend()
    plt.savefig('sigma_TPRS.png')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'sigma_FPRS.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(range(len(x)), y)
    plt.xticks(range(len(x)), x)
    plt.legend()
    plt.savefig(os.path.join(pre, 'sigma_FPRS.png'))
    plt.clf()


def plot_ratio(dataset_name):
    x = [0.01, 0.05, 0.1, 0.5, 1]
    pre = os.path.join(os.path.join('results', dataset_name))

    with open(os.path.join(pre, os.path.join('proposed', 'ratio_TPRS.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(range(len(x)), y)
    plt.xticks(range(len(x)), x)
    plt.legend()
    plt.savefig('ratio_TPRS.png')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'ratio_FPRS.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(range(len(x)), y)
    plt.xticks(range(len(x)), x)
    plt.legend()
    plt.savefig(os.path.join(pre, 'ratio_FPRS.png'))
    plt.clf()


def plot_time(dataset_name, ed):
    x = range(ed)
    pre = os.path.join(os.path.join('results', dataset_name))

    with open(os.path.join(pre, os.path.join('proposed', 'TPRS.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(x, y)
    plt.legend()
    plt.savefig('TPRS.png')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'FPRS.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(x, y)
    plt.legend()
    plt.savefig('FPRS.png')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'RSE.json')), 'r') as FR:
        y = json.loads(FR.read())

    plt.plot(x, y)
    plt.legend()
    plt.savefig('RSE.png')
    plt.clf()


def locations():
    x = range(ed)
    pre = os.path.join(os.path.join('results', dataset_name))
    with open(os.path.join(pre, os.path.join('proposed', 'false_locations.json')), 'r') as FR:
        y = json.loads(FR.read())

    st = {}
    for i in range(len(y)):
        for ele in range(y[i]):
            st[str(ele[0]) + ' ' + str(ele[1])] += 1

    for key in st:
        print(key+':'+st[key])
    # plt.plot(x, y)
    # plt.legend()
    # plt.savefig('RSE.png')
    # plt.clf()


if __name__ == '__main__':
    print("changed")
    locations()