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

def plot_final_outlier_ratio(dataset_name):
    x = []
    for i in range(10):
        x.append(i / 10)

    plt.figure(figsize=(14, 6))

    # T-online noise_scheme: Gaussian, outliers_scheme: random
    y1 = [0.9771760416666673, 0.9749359623015879, 0.9769278291153296, 0.9773115793928305, 0.9790448734272263, 0.9805892285211996, 0.9746154576654011, 0.9808633742026612, 0.9780628426534764, 0.9785193059878463]
    
    # T-online noise_scheme: Gaussian, outliers_scheme: Exponential
    y2 = [0.9491023809523833, 0.9343518308080863, 0.9307166278166317, 0.9315883318070803, 0.9332154612243614, 0.9339241748163037, 0.9312864307078829, 0.9324700487550633, 0.9317004570173697, 0.9322872729337796]
    
    # T-online noise_scheme: Gaussian, outliers_scheme: Gaussian
    y3 = [0.9338175595238121, 0.9244026289682604, 0.9206369318181868, 0.9203847422389722, 0.9209521860288358, 0.9206287494067682, 0.9242320316885139, 0.9233479688940006, 0.924069788374433, 0.923719626242013]

    # T-online noise_scheme: Gaussian, outliers_scheme: structural

    y4 = [0.978375, 0.9745, 0.97771875, 0.9774500000000009, 0.9779107142857192, 0.9744375, 0.9783749999999956, 0.9774318181818257, 0.9760000000000089, 0.9782053571428603]

    # VITAD noise_scheme: Gaussian, outliers_scheme: random
    y5 = [0.992940, 0.991720, 0.992262, 0.986436, 1, 1, 1, 1, 1, 1]
    # VITAD noise_scheme: Gaussian, outliers_scheme: Exponential
    y6 = [0.957606, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # VITAD noise_scheme: Gaussian, outliers_scheme: Gaussian
    y7 = [0.943630, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # VITAD noise_scheme: Gaussian, outliers_scheme: structural
    y8 = [0.993625, 0.991250, 0.978406, 0.993550, 1, 1, 1, 1, 1, 1]


    plt.subplot(1, 2, 1)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='T-online noise: Gaussian, outliers: random')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='T-online noise: Gaussian, outliers: Exponential')
    plt.plot(range(len(x)), y3, color='#aa5598', marker='v', linestyle='--', label='T-online noise: Gaussian, outliers: Gaussian')
    plt.plot(range(len(x)), y4, color='#32509c', marker='p', linestyle='--', label='T-online noise: Gaussian, outliers: structural')
    plt.plot(range(len(x)), y5, color='#d93731', marker='*', linestyle='--', label='VITAD noise: Gaussian, outliers: random')
    plt.plot(range(len(x)), y6, color='#ddccc5', marker='D', linestyle='-.', label='VITAD noise: Gaussian, outliers: Exponential')
    plt.plot(range(len(x)), y7, color='#ffa289', marker='h', linestyle=':', label='VITAD noise: Gaussian, outliers: Gaussian')
    plt.plot(range(len(x)), y8, color='#4ec9b0', marker='X', linestyle=':', label='VITAD noise: Gaussian, outliers: structural')
    plt.xticks(range(len(x)), x)
    plt.xlabel('outlier ratio', fontsize=14)
    plt.ylabel('TPR', fontsize=14)

    # T-online noise_scheme: Gaussian, outliers_scheme: random
    y1 = [0.0002859621738373123, 0.0005582550932535048, 0.0007550453269559344, 0.0009650747645486992, 0.0011470933312162435, 0.0012918660652188074, 0.0019597030364725118, 0.001730272392071036, 0.0022279514481818286, 0.0024371587031501176]

    # T-online noise_scheme: Gaussian, outliers_scheme: Exponential
    y2 = [0.000679936335132494, 0.001418785037948584, 0.0021671671067188497, 0.0029018074012744483, 0.003591146188892552, 0.004294291792242073, 0.005222203428427119, 0.005904113085795402, 0.00678964181281522, 0.007590888283783878]
    
    # T-online noise_scheme: Gaussian, outliers_scheme: Gaussian
    y3 = [0.0008633685581869231, 0.0016827190589197968, 0.0025005080764070814, 0.003337741572618528, 0.004149335946214509, 0.00505965148763266, 0.0057075281004491385, 0.006800150262737173, 0.007529068571119375, 0.008541115026912578]

    # T-online noise_scheme: Gaussian, outliers_scheme: structural
    y4 = [0.00015122377622377642, 0.0003591549295774676, 0.0006366071428571408, 0.0008111510791366861, 0.0011286496350365164, 0.0015036764705882789, 0.0016138059701492741, 0.0018665413533835058, 0.0021818181818182296, 0.0023471153846154526]

    # VITAD noise_scheme: Gaussian, outliers_scheme: random
    y5 = [0.000092, 0.000192, 0.000262, 0.000573, 0, 0, 0, 0, 0, 0]
    # VITAD noise_scheme: Gaussian, outliers_scheme: Exponential
    y6 = [0.000571, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # VITAD noise_scheme: Gaussian, outliers_scheme: Gaussian
    y7 = [0.0007310, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # VITAD noise_scheme: Gaussian, outliers_scheme: structural
    y8 = [0.000045, 0.000123, 0.000617, 0.000232, 0, 0, 0, 0, 0, 0]

    plt.subplot(1, 2, 2)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='T-online noise: Gaussian, outliers: random')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='T-online noise: Gaussian, outliers: Exponential')
    plt.plot(range(len(x)), y3, color='#aa5598', marker='v', linestyle='--', label='T-online noise: Gaussian, outliers: Gaussian')
    plt.plot(range(len(x)), y4, color='#32509c', marker='p', linestyle='--', label='T-online noise: Gaussian, outliers: structural')
    plt.plot(range(len(x)), y5, color='#d93731', marker='*', linestyle='--', label='VITAD noise: Gaussian, outliers: random')
    plt.plot(range(len(x)), y6, color='#ddccc5', marker='D', linestyle='-.', label='VITAD noise: Gaussian, outliers: Exponential')
    plt.plot(range(len(x)), y7, color='#ffa289', marker='h', linestyle=':', label='VITAD noise: Gaussian, outliers: Gaussian')
    plt.plot(range(len(x)), y8, color='#4ec9b0', marker='X', linestyle=':', label='VITAD noise: Gaussian, outliers: structural')
    plt.xticks(range(len(x)), x)
    plt.legend(fontsize=12)
    plt.xlabel('outlier ratio', fontsize=14)
    plt.ylabel('FPR', fontsize=14)
    plt.savefig('outlier.png', format='png')
    plt.savefig('outlier.pdf', format='pdf')
    plt.savefig('outlier.eps', format='eps')
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
    # plot_noise('Abilene')
    plot_final_outlier_ratio('Abilene')