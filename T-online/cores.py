from T_online import *
sys.path.append("../")
from utils import *
Rs = {'Abilene': 12, 'GEANT': 12, 'CERNET': 12, 'VData50': 12, 'VData100': 12, 'VData200': 12, 'VData300': 12, 'VData400': 12, 'VData500': 12}
Ws = {'Abilene': 2016, 'GEANT': 672, 'CERNET': 2016, 'VData50': 224, 'VData100': 224, 'VData200': 224, 'VData300': 224, 'VData400': 224, 'VData500': 224}


def evaluate(dataset_name, parameters):
    # Run BayesCP
    ed = parameters['ed']
    R = parameters['R']
    init = parameters['init']
    Y, outliers_p = generator(dataset_name, parameters)

    if ed == None:
        ed = Y.shape[2]
    Y = Y[:, :, 0:ed]
    outliers_p = outliers_p[:, :, 0:ed]
    L, S = T_online(Y=Y, epsilon=np.sum(outliers_p, (0, 1)), R=R, W=Ws[dataset_name], maxiters=20, init=init)
    model = {}
    model['TPRS'] = []
    model['FPRS'] = []
    model['precision'] = 0
    model['FPR'] = 0
    for i in range(Y.shape[2]):
        # [TPR, FPR, false_locations] = check(S[:, :, i], outliers_p[:, :, i], np.sum(outliers_p[:, :, i]), Y.shape)
        [TPR, FPR] = check(S[:, :, i], outliers_p[:, :, i], np.sum(outliers_p[:, :, i]), Y.shape)
        model['TPRS'].append(TPR)
        model['FPRS'].append(FPR)
        model['precision'] += TPR
        model['FPR'] += FPR
    model['precision'] /= Y.shape[2]
    model['FPR'] /= Y.shape[2]
    return model


def eval_time(dataset_name, ed):
    # Run BayesCP
    model = evaluate(dataset_name, ed, 0.1, 0, 0.1, 15)
    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/TPRS.json'), 'w') as FD:
        FD.write(json.dumps(model['TPRS']))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/FPRS.json'), 'w') as FD:
        FD.write(json.dumps(model['FPRS']))


def eval_R(dataset_name):
    TPRS = [[] for _ in range(2, 51, 2)]
    FPRS = [[] for _ in range(2, 51, 2)]
    for i in range(10):
        for R in range(2, 51, 2):
            model = evaluate(dataset_name, 8000, 0.1, 0, 0.1, R)
            TPRS[i].append(model['precision'])
            FPRS[i].append(model['FPR'])

    TPRS_mean = []
    FPRS_mean = []
    for i in range(len(range(2, 51, 2))):
        sum = 0
        sum2 = 0
        for j in range(10):
            sum += TPRS[j][i]
            sum2 += FPRS[j][i]
        TPRS_mean.append(sum / 10)
        FPRS_mean.append(sum2 / 10)

    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/R_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS_mean))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/R_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS_mean))


def eval_mu(dataset_name, ed):
    TPRS = []
    FPRS = []
    mus = [0.01, 0.05, 0.1, 0.5, 1]
    for mu in mus:
        model = evaluate(dataset_name, ed, 0.1, mu, 0.1, Rs[dataset_name])
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/mu_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/mu_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_sigma(dataset_name):
    TPRS = []
    FPRS = []
    sigmas = [0.01, 0.05, 0.1, 0.5, 1]
    for sigma in sigmas:
        model = evaluate(dataset_name, 8000, 0.1, 0, sigma, Rs[dataset_name])
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/sigma_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/sigma_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_ratio(dataset_name, parameters):
    TPRS = []
    FPRS = []

    for fraction in range(1, 6, 1):
        parameters['fraction'] = fraction / 100
        start = time.time()
        model = evaluate(dataset_name, parameters)
        end = time.time()
        logging.info("one loop cost %f:" %(end - start)/60)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    TPR_file_path = 'T-online/ratio'
    FPR_file_path = 'T-online/ratio'
    if parameters['noise_scheme'] != None:
        TPR_file_path += '_' + parameters['noise_scheme']
        FPR_file_path += '_' + parameters['noise_scheme']

    if parameters['outliers_scheme'] != None:
        TPR_file_path += '_' + parameters['outliers_scheme']
        FPR_file_path += '_' + parameters['outliers_scheme']

    TPR_file_path += '_TPRS.json'
    FPR_file_path += '_FPRS.json'
    with open(os.path.join(os.path.join('../../results', dataset_name), TPR_file_path), 'w') as FD:
        logging.info("write to file %s:" %(TPR_file_path))
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), FPR_file_path), 'w') as FD:
        logging.info("write to file %s:" %(FPR_file_path))
        FD.write(json.dumps(FPRS))


def eval_ratio2(dataset_name, init="random"):
    print('eval_ratio2')
    TPRS = []
    FPRS = []
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    for fraction in fractions:
        model = evaluate(dataset_name, 8000, fraction, 0, 0.1, Rs[dataset_name], distribution="Gaussian", init=init)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/ratio2_'+"Gaussian"+'_'+init+'_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/ratio2_'+"Gaussian"+'_'+init+'_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_SNR(dataset_name, distribution="Gaussian", init="random"):
    TPRS = []
    FPRS = []
    SNRs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
    for SNR in SNRs:
        model = evaluate(dataset_name, 8000, 0.1, 0, 0.1, Rs[dataset_name], SNR=SNR, distribution=distribution, init=init)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])
        print(SNR)

    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/SNR_'+distribution+'_'+init+'_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'T-online/SNR_'+distribution+'_'+init+'_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


if __name__ == '__main__':
    Y, outliers_p = generator("CERNET", 0.1, 0.05, 0.1, distribution="Gaussian", SNR=None)

    Y = Y[:, :, 0:8000]
    start = time.time()
    kruskal_tensor = parafac(Y, rank=12, n_iter_max=15, init="svd")
    end = time.time()
    print(' consumption :', end - start)