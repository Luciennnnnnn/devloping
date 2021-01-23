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

def plot_noise(dataset_name):
    x = []
    for i in range(10):
        x.append(i / 10)
    pre = os.path.join(os.path.join('results', dataset_name))

    # with open(os.path.join(pre, os.path.join('proposed', 'noise_ER.json')), 'r') as FR:
    #     y = json.loads(FR.read())

    # with open(os.path.join(pre, os.path.join('proposed', 'noise_ER.json')), 'r') as FR:
    #     y = json.loads(FR.read())

    # with open(os.path.join(pre, os.path.join('proposed', 'noise_ER.json')), 'r') as FR:
    #     y = json.loads(FR.read())

    # with open(os.path.join(pre, os.path.join('proposed', 'noise_ER.json')), 'r') as FR:
    #     y = json.loads(FR.read())

    # with open(os.path.join(pre, os.path.join('proposed', 'noise_ER.json')), 'r') as FR:
    #     y = json.loads(FR.read())
    plt.figure(figsize=(14, 6))
    # Gaussian anomaly enum anomaly ratio
    # [1.218317, 1.231309, 1.201047, 1.219431, 1.131734, 1.171725]

    # ER: Without noise and anomaly enum missing ratio from 0 ~ 0.9 Done 2021-01-21 22:30:36.log
    y1 = [1.257989, 1.479484, 1.520001, 2.118648, 2.348242, 3.571217, 3.374463, 12.244402, 49.602915, 295.510673]
    
    # ER: Gaussian noise enum missing ratio from 0 ~ 0.9 Done 2021-01-21 22:32:26.log
    y2 = [1.174613, 1.301723, 1.533960, 1.977712, 2.038753, 3.141521, 4.522832, 9.745681, 28.832551, 338.805911]
    
    # ER: Gaussian anomaly enum missing ratio from 0 ~ 0.9 Done 2021-01-21 22:51:40.log
    y3 = [1.135438, 1.285860, 1.441335, 1.787816, 1.847997, 2.851324, 4.331003, 12.189432, 28.037928, 438.592552]
    
    # ER: Random anomaly enum missing ratio from 0 ~ 0.9 17225 2021-01-22 22:07:58.log
    y4 = np.random.randn(10) # 很大 考虑RSE

    # ER: Structural anomaly enum missing ratio from 0 ~ 0.9 17196 2021-01-22 22:01:26.log
    y5 = np.random.randn(10) # 很大 考虑RSE

    # ER: Gaussian noise and Random anomaly enum missing ratio from 0 ~ 0.9 17250 2021-01-22 22:11:16.log
    y6 = np.random.randn(10) # 很怪 貌似计算进了噪声 打印看看

    # ER: Gaussian noise and Exponential anomaly enum missing ratio from 0 ~ 0.9 17291 2021-01-22 22:32:02.log
    y7 = np.random.randn(10) # 正常

    # ER: Gaussian noise and Structural anomaly enum missing ratio from 0 ~ 0.9 ?? 2021-01-21 22:50:44.log 好 ｜｜ 2021-01-22 09:42:32.log 坏
    y8 = [1.187148, 1.381492, 1.499855, 2.084450, 1.653089, 2.005807, 3.901273, 11, 11, 11]
    plt.subplot(1, 2, 1)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='Without noise and anomaly')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='Gaussian noise')
    plt.plot(range(len(x)), y3, color='#aa5598', marker='v', linestyle='--', label='Gaussian anomaly')
    plt.plot(range(len(x)), y4, color='#32509c', marker='p', linestyle='--', label='Random anomaly')
    plt.plot(range(len(x)), y5, color='#d93731', marker='*', linestyle='--', label='Structural anomaly')
    plt.plot(range(len(x)), y6, color='#ddccc5', marker='D', linestyle='-.', label='Gaussian noise and Random anomaly')
    plt.plot(range(len(x)), y7, color='#ffa289', marker='h', linestyle=':', label='Gaussian noise and Exponential anomaly')
    plt.plot(range(len(x)), y8, color='#4ec9b0', marker='X', linestyle=':', label='Gaussian noise and Structural anomaly')
    plt.xticks(range(len(x)), x)
    plt.legend(fontsize=14)
    plt.xlabel('Missing ratio', fontsize=14)
    plt.ylabel('ER', fontsize=14)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='Without noise and anomaly')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='Gaussian noise')
    plt.plot(range(len(x)), y3, color='#aa5598', marker='v', linestyle='--', label='Gaussian anomaly')
    plt.plot(range(len(x)), y4, color='#32509c', marker='p', linestyle='--', label='Random anomaly')
    plt.plot(range(len(x)), y5, color='#d93731', marker='*', linestyle='--', label='Structural anomaly')
    plt.plot(range(len(x)), y6, color='#ddccc5', marker='D', linestyle='-.', label='Gaussian noise and Random anomaly')
    plt.plot(range(len(x)), y7, color='#ffa289', marker='h', linestyle=':', label='Gaussian noise and Exponential anomaly')
    plt.plot(range(len(x)), y8, color='#4ec9b0', marker='X', linestyle=':', label='Gaussian noise and Structural anomaly')
    plt.xticks(range(len(x)), x)
    plt.legend(fontsize=14)
    plt.xlabel('Missing ratio', fontsize=14)
    plt.ylabel('FPR', fontsize=14)
    plt.savefig('noise_ER.png', format='png')
    plt.savefig('noise_ER.pdf', format='pdf')
    plt.savefig('noise_ER.eps', format='eps')
    plt.clf()
    plt.close()

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
    plot_noise('Abilene')