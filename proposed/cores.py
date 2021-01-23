import logging
from proposed import *

Rs = {'Abilene': 2, 'GEANT': 12, 'CERNET': 6, 'VData50': 6, 'VData100': 6, 'VData200': 6, 'VData300': 6, 'VData400': 6, 'VData500': 6}
inits = {'Abilene': 'rand', 'GEANT': 'rand', 'CERNET': 'ml', 'VData50': 'rand', 'VData100': 'rand', 'VData200': 'rand', 'VData300': 'rand', 'VData400': 'rand', 'VData500': 'rand'}


def evaluate(dataset_name, parameters):
    # Run BayesCP
    Y, outliers, outliers_p, noises, Omega = generator(dataset_name, parameters)
    ed = parameters['ed']
    R = parameters['R']
    theta = parameters['theta']
    Y = Y[:, :, 0:ed]
    outliers = outliers[:, :, 0:ed]
    outliers_p = outliers_p[:, :, 0:ed]
    noises = noises[:, :, 0:ed]
    Omega = Omega[:, :, 0:ed]
    #noises = noises[:, :, 0:ed]
    # print('Omega', np.sum(Omega != 1))
    # print('outlier', np.sum(outliers_p * outlier))
    # print('noises', np.sum(noises))
    model = VITAD(Y=Y + outliers + noises, outliers_p=outliers_p, Omega=Omega, maxRank=R, maxiters=20, tol=1e-4, init=inits[dataset_name])
    model['RSE'] = norm(model['X'] - Y) / norm(Y)
    model['ER'] = np.sum(np.square(model['X'] - Y)) / np.sum(np.square(Y))
    model['SRR'] = np.sum(np.abs((model['X'] - Y) / Y) <= theta) / np.prod(Y.shape)

    logging.debug("noise: %s" %(str(noises)))
    logging.debug("Y: %s" %(str(Y)))
    logging.debug("X: %s" %(str(model['X2'])))
    logging.debug("Y + noise: %s" %(str(Y + noises)))

    logging.debug("RSE2 %f:" %(norm(model['X'] - Y) / norm(Y)))
    logging.debug("ER2 %f:" %(np.sum(np.square(model['X2'] - Y)) / np.sum(np.square(Y))))
    logging.debug("SRR2 %f:" %(np.sum(np.abs((model['X2'] - Y) / Y) <= theta) / np.prod(Y.shape)))
    logging.debug("SRR22 %f:" %( np.sum((model['X2'] - Y) / Y <= theta) / np.prod(Y.shape) ) )
    
    Y += noises
    logging.debug("@ER2 %f:" %(np.sum(np.square(model['X2'] - Y)) / np.sum(np.square(Y))))
    logging.debug("@SRR2 %f:" %(np.sum(np.abs((model['X2'] - Y) / Y) <= theta) / np.prod(Y.shape)))
    logging.debug("@SRR22 %f:" %( np.sum((model['X2'] - Y) / Y <= theta) / np.prod(Y.shape) ) )
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


def eval_R(dataset_name, parameters):
    nums = 1
    up = 15
    TPRS = [[] for _ in range(2, up, 2)]
    FPRS = [[] for _ in range(2, up, 2)]
    for i in range(nums):
        for R in range(2, up, 2):
            parameters['R'] = R
            start = time.time()
            model = evaluate(dataset_name, parameters)
            end = time.time()
            logging.info("dataset: %s" %(dataset_name))
            logging.info("one loop cost %f:" %((end - start)/60))
            logging.debug("Rank: %d, TPR %f:" %(parameters['R'], model['precision']))
            logging.debug("Rank: %d, FPR %f:" %(parameters['R'], model['FPR']))
            logging.debug("Rank: %d, ER %f:" %(parameters['R'], model['ER']))
            logging.debug("Rank: %d, SRR %f:" %(parameters['R'], model['SRR']))
            TPRS[i].append(model['precision'])
            FPRS[i].append(model['FPR'])

    TPRS_mean = []
    FPRS_mean = []
    for i in range(len(range(2, up, 2))):
        sum = 0
        sum2 = 0
        for j in range(nums):
            sum += TPRS[j][i]
            sum2 += FPRS[j][i]
        TPRS_mean.append(sum / nums)
        FPRS_mean.append(sum2 / nums)

    TPR_file_path = 'proposed/R'
    FPR_file_path = 'proposed/R'
    if parameters['noise_scheme'] != None:
        TPR_file_path += '_' + parameters['noise_scheme']
        FPR_file_path += '_' + parameters['noise_scheme']

    if parameters['outliers_scheme'] != None:
        TPR_file_path += '_' + parameters['outliers_scheme']
        FPR_file_path += '_' + parameters['outliers_scheme']

    TPR_file_path += '_TPRS.json'
    FPR_file_path += '_FPRS.json'

    if not os.path.exists(os.path.dirname(os.path.join(os.path.join('../../results', dataset_name), TPR_file_path))):
        os.makedirs(os.path.dirname(os.path.join(os.path.join('../../results', dataset_name), TPR_file_path)))

    with open(os.path.join(os.path.join('../../results', dataset_name), TPR_file_path), 'w') as FD:
        FD.write(json.dumps(TPRS_mean))
        logging.info("write to file %s:" %(TPR_file_path))
    with open(os.path.join(os.path.join('../../results', dataset_name), FPR_file_path), 'w') as FD:
        FD.write(json.dumps(FPRS_mean))
        logging.info("write to file %s:" %(FPR_file_path))


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
        start = time.time()
        model = evaluate(dataset_name, parameters)
        end = time.time()
        logging.info("dataset: %s, fraction: %f" %(dataset_name, parameters['fraction']))
        logging.info("one loop cost %f:" %((end - start)/60))
        logging.debug("TPR %f:" %(model['precision']))
        logging.debug("FPR %f:" %(model['FPR']))
        logging.debug("ER %f:" %(model['ER']))
        logging.debug("SRR %f:" %(model['SRR']))
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])
        #if fraction == 10:
        #    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/false_locations.json'), 'w') as FD:
        #        FD.write(json.dumps(model['false_locations']))
    TPR_file_path = 'proposed/ratio'
    FPR_file_path = 'proposed/ratio'
    if parameters['noise_scheme'] != None:
        TPR_file_path += '_' + parameters['noise_scheme']
        FPR_file_path += '_' + parameters['noise_scheme']

    if parameters['outliers_scheme'] != None:
        TPR_file_path += '_' + parameters['outliers_scheme']
        FPR_file_path += '_' + parameters['outliers_scheme']

    TPR_file_path += '_TPRS.json'
    FPR_file_path += '_FPRS.json'

    if not os.path.exists(os.path.dirname(os.path.join(os.path.join('../../results', dataset_name), TPR_file_path))):
        os.makedirs(os.path.dirname(os.path.join(os.path.join('../../results', dataset_name), TPR_file_path)))
        
    with open(os.path.join(os.path.join('../../results', dataset_name), TPR_file_path), 'w') as FD:
        logging.info("write to file %s:" %(TPR_file_path))
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), FPR_file_path), 'w') as FD:
        logging.info("write to file %s:" %(FPR_file_path))
        FD.write(json.dumps(FPRS))

def eval_missing_ratio(dataset_name, parameters):
    TPRS = []
    FPRS = []
    ERS = []
    SRRS = []
    for fraction in range(0, 10):
        parameters['missing_ratio'] = fraction / 10
        start = time.time()
        model = evaluate(dataset_name, parameters)
        end = time.time()
        logging.info("dataset: %s, missing_ratio: %s" %(dataset_name, parameters['missing_ratio']))
        logging.info("one loop cost %f:" %((end - start)/60))
        logging.debug("TPR %f:" %(model['precision']))
        logging.debug("FPR %f:" %(model['FPR']))
        logging.debug("ER %f:" %(model['ER']))
        logging.debug("SRR %f:" %(model['SRR']))
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])
        ERS.append(model['ER'])
        SRRS.append(model['SRR'])
        #if fraction == 10:
        #    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/false_locations.json'), 'w') as FD:
        #        FD.write(json.dumps(model['false_locations']))
    TPR_file_path = 'proposed/missing_ratio'
    FPR_file_path = 'proposed/missing_ratio'
    ER_file_path = 'proposed/missing_ratio'
    SRR_file_path = 'proposed/missing_ratio'
    if parameters['noise_scheme'] != None:
        TPR_file_path += '_' + parameters['noise_scheme']
        FPR_file_path += '_' + parameters['noise_scheme']
        ER_file_path += '_' + parameters['noise_scheme']
        SRR_file_path += '_' + parameters['noise_scheme']


    if parameters['outliers_scheme'] != None:
        TPR_file_path += '_' + parameters['outliers_scheme']
        FPR_file_path += '_' + parameters['outliers_scheme']
        ER_file_path += '_' + parameters['outliers_scheme']
        SRR_file_path += '_' + parameters['outliers_scheme']

    TPR_file_path += '_TPRS.json'
    FPR_file_path += '_FPRS.json'
    ER_file_path += '_ERS.json'
    SRR_file_path += '_SRRS.json'

    if not os.path.exists(os.path.dirname(os.path.join(os.path.join('../../results', dataset_name), TPR_file_path))):
        os.makedirs(os.path.dirname(os.path.join(os.path.join('../../results', dataset_name), TPR_file_path)))
        
    with open(os.path.join(os.path.join('../../results', dataset_name), TPR_file_path), 'w') as FD:
        logging.info("write to file %s:" %(TPR_file_path))
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), FPR_file_path), 'w') as FD:
        logging.info("write to file %s:" %(FPR_file_path))
        FD.write(json.dumps(FPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), ER_file_path), 'w') as FD:
        logging.info("write to file %s:" %(ER_file_path))
        FD.write(json.dumps(ERS))
    with open(os.path.join(os.path.join('../../results', dataset_name), SRR_file_path), 'w') as FD:
        logging.info("write to file %s:" %(SRR_file_path))
        FD.write(json.dumps(SRRS))

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