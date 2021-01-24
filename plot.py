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

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
    # Gaussian anomaly enum anomaly ratio
    # [1.218317, 1.231309, 1.201047, 1.219431, 1.131734, 1.171725]

    # ER: Without noise and anomaly enum missing ratio from 0 ~ 0.9 Done 2021-01-21 22:30:36.log
    y1 = [1.257989, 1.479484, 1.520001, 2.118648, 2.348242, 3.571217, 3.6122427376227653, 12.244402, 49.602915, 295.510673]
    # ER: Gaussian noise enum missing ratio from 0 ~ 0.9 Done 2021-01-21 22:32:26.log
    y2 = [1.174613, 1.301723, 1.533960, 1.977712, 2.038753, 3.141521, 4.522832, 9.745681, 28.832551, 338.805911]
    
    # ER: Gaussian anomaly enum missing ratio from 0 ~ 0.9 Done 2021-01-21 22:51:40.log
    y3 = [1.135438, 1.285860, 1.441335, 1.787816, 1.847997, 2.851324, 4.331003, 12.189432, 28.037928, 438.592552]
    
    # ER: Random anomaly enum missing ratio from 0 ~ 0.9 17225 2021-01-22 22:07:58.log
    y4 = np.random.randn(10) # 考虑RSE

    # ER: Structural anomaly enum missing ratio from 0 ~ 0.9 17196 2021-01-22 22:01:26.log
    y5 = np.random.randn(10) #考虑RSE

    # ER: Gaussian noise and Random anomaly enum missing ratio from 0 ~ 0.9 17250 2021-01-22 22:11:16.log
    y6 = np.random.randn(10) # 貌似计算进了噪声 打印看看

    # ER: Gaussian noise and Exponential anomaly enum missing ratio from 0 ~ 0.9 17291 2021-01-22 22:32:02.log
    y7 = np.random.randn(10) # 正常

    # ER: Gaussian noise and Structural anomaly enum missing ratio from 0 ~ 0.9 ?? 2021-01-21 22:50:44.log 好 ｜｜ 2021-01-22 09:42:32.log 坏
    y8 = [1.187148, 1.381492, 1.499855, 2.084450, 1.653089, 2.005807, 3.901273, 10.12415, 31.9345934, 442.094751134]

    # ER: noise: random, outliers: Gaussian
    y9 = [1.135438, 2.643356389832484, 3.3633011966147635, 6.247521371847992, 8.912397579723127, 11.322172433832524, 11.44767124476099, 23.90881119741411, 34.469985829338086, 451.2238542202693]
    
    # ER: noise: mixture, outliers: Gaussian
    y10 = [4.144085, 8.216227993035949, 15.487121229110633, 21.021754481598542, 23.789226084159907, 43.542224492537954, 50.5487106933616, 68.05000722648721, 217.4876030362145, 907.697348722337]
    
    ax1.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='without noise and outliers')
    ax1.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='noise only : Gaussian')
    ax1.plot(range(len(x)), y3, color='#aa5598', marker='v', linestyle='-.', label='outliers only : Gaussian')
    #ax1.plot(range(len(x)), y4, color='#32509c', marker='p', linestyle='--', label='Random anomaly')
    #ax1.plot(range(len(x)), y5, color='#d93731', marker='*', linestyle='--', label='Structural anomaly')
    #ax1.plot(range(len(x)), y6, color='#ddccc5', marker='D', linestyle='-.', label='Gaussian noise and Random anomaly')
    #ax1.plot(range(len(x)), y7, color='#ffa289', marker='h', linestyle=':', label='Gaussian noise and Exponential anomaly')
    ax1.plot(range(len(x)), y8, color='#4ec9b0', marker='X', linestyle=':', label='noise: Gaussian, outliers: Structural')
    ax1.plot(range(len(x)), y9, color='#32509c', marker='*', linestyle='-', label='noise: random, outliers: Gaussian')
    ax1.plot(range(len(x)), y10, color='#d93731', marker='p', linestyle='--', label='noise: mixture, outliers: Gaussian')
    ax1.set_xticks(range(len(x)))
    ax1.set_xticklabels(x)
    ax1.legend(fontsize=12)
    ax1.set_xlabel('Missing ratio', fontsize=14)
    ax1.set_ylabel('ER', fontsize=14)

    fig.savefig('noise_ER.png', format='png')
    fig.savefig('noise_ER.pdf', format='pdf')
    fig.savefig('noise_ER.eps', format='eps')
    plt.clf()

def plot_final_outlier_ratio(dataset_name):
    x = []
    for i in range(1, 11, 1):
        x.append(i / 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # T-online noise_scheme: Gaussian, outliers_scheme: random
    y1 = [0.9771760416666673, 0.9749359623015879, 0.9742097784374412, 0.9741680553246821, 0.9738162594966712, 0.9730777380368685, 0.9727194839392058, 0.9723453550963479, 0.9720831323701619, 0.9716099594696959]
    # T-online noise_scheme: Gaussian, outliers_scheme: Exponential
    y2 = [0.9491023809523833, 0.9343518308080863, 0.9307166278166317, 0.9306680764556164, 0.9306369419915795, 0.9304276776618011, 0.9303885823117126, 0.9300458400840629, 0.9300289677450645, 0.9296743068850877]
    # T-online noise_scheme: Gaussian, outliers_scheme: Gaussian
    y3 = [0.9338175595238121, 0.9244026289682604, 0.9206369318181868, 0.9203847422389722, 0.9201848389118741, 0.9199866636480947, 0.919899990334069, 0.9194328203243687, 0.9190659705137403, 0.9190315145345962]
    # T-online noise_scheme: Gaussian, outliers_scheme: structural
    y4 = [0.978375, 0.9745, 0.9714219127733846, 0.9675733426158918, 0.967494028003294, 0.9671377589476288, 0.9629845464003776, 0.9613980332731312, 0.9582780399108867, 0.958187222217241]
    # VITAD noise_scheme: Gaussian, outliers_scheme: random
    y5 = [0.99294, 0.99172, 0.9916242761898896, 0.986436, 0.9862327511364953, 0.9861473481455071, 0.9857588223287094, 0.9857021500439591, 0.9853446486288195, 0.9852310665167233]
    # VITAD noise_scheme: Gaussian, outliers_scheme: Exponential
    y6 = [0.957606, 0.948779, 0.947458, 0.9472097533063177, 0.9468964459701371, 0.946024073476967, 0.9451314323203485, 0.9442380966111049, 0.9433516833021884, 0.9429028043970835]
    # VITAD noise_scheme: Gaussian, outliers_scheme: Gaussian
    y7 = [0.94363, 0.932196, 0.932024, 0.9310814430707657, 0.9303462168250825, 0.9299483249539463, 0.9296982177995623, 0.9289666831764164, 0.9283323826506614, 0.9274057196984615]
    # VITAD noise_scheme: Gaussian, outliers_scheme: structural
    y8 = [0.993625, 0.99125, 0.978406, 0.9781993003232978, 0.9777227222997095, 0.9772698975852336, 0.9772362918664942, 0.9772253852481262, 0.9768678579415543, 0.9767989554410533]

    ax1.plot(range(len(x)), y1, color='#3d5d46', marker='1', linestyle='--', label='T-online - noise: Gaussian, outliers: random')
    ax1.plot(range(len(x)), y2, color='#8dabb6', marker='2', linestyle='--', label='T-online - noise: Gaussian, outliers: Exponential')
    ax1.plot(range(len(x)), y3, color='#aa5598', marker='3', linestyle='--', label='T-online - noise: Gaussian, outliers: Gaussian')
    ax1.plot(range(len(x)), y4, color='#32509c', marker='4', linestyle='--', label='T-online - noise: Gaussian, outliers: structural')
    ax1.plot(range(len(x)), y5, color='#d93731', marker='v', linestyle='-', label='VITAD - noise: Gaussian, outliers: random')
    ax1.plot(range(len(x)), y6, color='#ddccc5', marker='^', linestyle='-', label='VITAD - noise: Gaussian, outliers: Exponential')
    ax1.plot(range(len(x)), y7, color='#ffa289', marker='<', linestyle='-', label='VITAD - noise: Gaussian, outliers: Gaussian')
    ax1.plot(range(len(x)), y8, color='#4ec9b0', marker='>', linestyle='-', label='VITAD - noise: Gaussian, outliers: structural')
    ax1.set_xticks(range(len(x)))
    ax1.set_xticklabels(x)
    ax1.set_xlabel('outlier ratio', fontsize=14)
    ax1.set_ylabel('TPR', fontsize=14)
    ax1.text(0.5, -0.135, '(a)', ha='center', fontsize=14, weight='bold', transform=ax1.transAxes)

    # T-online noise_scheme: Gaussian, outliers_scheme: random
    y1 = [0.0002859621738373123, 0.0005582550932535048, 0.0007550453269559344, 0.0009650747645486992, 0.0011470933312162435, 0.0012918660652188074, 0.0019597030364725118, 0.002237398826330531, 0.002343284481953799, 0.0024371587031501176]
    # T-online noise_scheme: Gaussian, outliers_scheme: Exponential
    y2 = [0.000679936335132494, 0.001418785037948584, 0.0021671671067188497, 0.0029018074012744483, 0.003591146188892552, 0.004294291792242073, 0.005222203428427119, 0.005904113085795402, 0.00678964181281522, 0.007590888283783878]
    # T-online noise_scheme: Gaussian, outliers_scheme: Gaussian
    y3 = [0.0008633685581869231, 0.0016827190589197968, 0.0025005080764070814, 0.003337741572618528, 0.004149335946214509, 0.00505965148763266, 0.0057075281004491385, 0.006800150262737173, 0.007529068571119375, 0.008541115026912578]
    # T-online noise_scheme: Gaussian, outliers_scheme: structural
    y4 = [0.00015122377622377642, 0.0003591549295774676, 0.0006366071428571408, 0.0008111510791366861, 0.0011286496350365164, 0.0015036764705882789, 0.0016138059701492741, 0.0018665413533835058, 0.0021818181818182296, 0.0023471153846154526]
    # VITAD noise_scheme: Gaussian, outliers_scheme: random
    y5 = [9.2e-05, 0.000192, 0.000262, 0.000573, 0.0009356831143836569, 0.0010432037287413517, 0.0012071354414814723, 0.0014476572226113525, 0.0016137874942879276, 0.0017122776749523482]
    # VITAD noise_scheme: Gaussian, outliers_scheme: Exponential
    y6 = [0.000571, 0.00111, 0.001651, 0.00177222985827452, 0.0022661992270921797, 0.0026879484379159265, 0.0027427961818496057, 0.00290185677110716, 0.003293573989168836, 0.0034975389012130514]
    # VITAD noise_scheme: Gaussian, outliers_scheme: Gaussian
    y7 = [0.000731, 0.001444, 0.002173, 0.0026264432474812993, 0.0027273129432005506, 0.003103584737284152, 0.0034797671009645127, 0.0038498213714798515, 0.004307588981246996, 0.004500311177559685]
    # VITAD noise_scheme: Gaussian, outliers_scheme: structural
    y8 = [4.5e-05, 0.000123, 0.000617, 0.0009129977980066233, 0.0010159151542109043, 0.0013869537525212324, 0.0017590321799327371, 0.001956229444830616, 0.0022901665328532, 0.0024567800548143796]

    ax2.plot(range(len(x)), y1, color='#3d5d46', marker='1', linestyle='--', label='T-online - noise: Gaussian, outliers: random')
    ax2.plot(range(len(x)), y2, color='#8dabb6', marker='2', linestyle='--', label='T-online - noise: Gaussian, outliers: Exponential')
    ax2.plot(range(len(x)), y3, color='#aa5598', marker='3', linestyle='--', label='T-online - noise: Gaussian, outliers: Gaussian')
    ax2.plot(range(len(x)), y4, color='#32509c', marker='4', linestyle='--', label='T-online - noise: Gaussian, outliers: structural')
    ax2.plot(range(len(x)), y5, color='#d93731', marker='v', linestyle='-', label='VITAD - noise: Gaussian, outliers: random')
    ax2.plot(range(len(x)), y6, color='#ddccc5', marker='^', linestyle='-', label='VITAD - noise: Gaussian, outliers: Exponential')
    ax2.plot(range(len(x)), y7, color='#ffa289', marker='<', linestyle='-', label='VITAD - noise: Gaussian, outliers: Gaussian')
    ax2.plot(range(len(x)), y8, color='#4ec9b0', marker='>', linestyle='-', label='VITAD - noise: Gaussian, outliers: structural')
    ax2.set_xticks(range(len(x)))
    ax2.set_xticklabels(x)
    ax2.legend(fontsize=12)
    ax2.set_xlabel('outlier ratio', fontsize=14)
    ax2.set_ylabel('FPR', fontsize=14)
    ax2.text(0.5, -0.135, '(b)', ha='center', fontsize=14, weight='bold', transform=ax2.transAxes)
    fig.savefig('outlier.png', format='png')
    fig.savefig('outlier.pdf', format='pdf')
    fig.savefig('outlier.eps', format='eps')
    fig.clf()

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
    # plot_final_outlier_ratio('Abilene')