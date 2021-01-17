from proposed import *

Rs = {'Abilene': 2, 'GEANT': 12, 'CERNET': 6, 'VData50': 6, 'VData100': 6, 'VData200': 6, 'VData300': 6, 'VData400': 6, 'VData500': 6}
inits = {'Abilene': 'rand', 'GEANT': 'rand', 'CERNET': 'ml', 'VData50': 'rand', 'VData100': 'rand', 'VData200': 'rand', 'VData300': 'rand', 'VData400': 'rand', 'VData500': 'rand'}


def evaluate(dataset_name, parameters):
    # Run BayesCP
    Y, outliers_p = generator(dataset_name, parameters)
    ed = parameters['ed']
    R = parameters['R']
    Y = Y[:, :, 0:ed]
    outliers_p = outliers_p[:, :, 0:ed]
    print('in')
    model = VITAD(Y=Y, outliers_p=outliers_p, maxRank=R, maxiters=20, tol=1e-4, verbose=True, init=inits[dataset_name])
    return model


def eval_time(dataset_name, ed):
    # Run BayesCP
    model = evaluate(dataset_name, ed, 0.1, 0, 0.1, Rs[dataset_name])
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/TPRS.json'), 'w') as FD:
        FD.write(json.dumps(model['TPRS']))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/FPRS.json'), 'w') as FD:
        FD.write(json.dumps(model['FPRS']))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/RSE.json'), 'w') as FD:
        FD.write(json.dumps(model['RSE']))


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

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/R_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS_mean))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/R_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS_mean))


def eval_mu(dataset_name, noise_scheme='outlier', ed=None, file_prefix='mu'):
    TPRS = []
    FPRS = []
    mus = [0.01, 0.05, 0.1, 0.5, 1]
    for mu in mus:
        model = evaluate(dataset_name, ed, 0.1, mu, 0.1, Rs[dataset_name], noise_scheme=noise_scheme)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/' + file_prefix + '_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/' + file_prefix + '_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_sigma(dataset_name):
    TPRS = []
    FPRS = []
    sigmas = [0.01, 0.05, 0.1, 0.5, 1]
    for sigma in sigmas:
        model = evaluate(dataset_name, 8000, 0.1, 0, sigma, Rs[dataset_name])
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/sigma_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/sigma_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_ratio(dataset_name, parameters):
    TPRS = []
    FPRS = []

    for fraction in range(1, 11, 1):
        parameters['fraction'] = fraction / 100
        model = evaluate(dataset_name, parameters)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])
        #if fraction == 10:
        #    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/false_locations.json'), 'w') as FD:
        #        FD.write(json.dumps(model['false_locations']))

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/ratio_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/ratio_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_ratio2(dataset_name):
    TPRS = []
    FPRS = []
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    for fraction in fractions:
        model = evaluate(dataset_name, 8000, fraction, 0, 0.1, Rs[dataset_name])
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/ratio2_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/ratio2_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_SNR(dataset_name):
    TPRS = []
    FPRS = []
    SNRs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
    for SNR in SNRs:
        model = evaluate(dataset_name, 8000, 0.1, 0, 0.1, Rs[dataset_name], SNR)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])
        print(SNR)

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/SNR1_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/SNR1_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))