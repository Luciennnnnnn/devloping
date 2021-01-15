from GraphTR2 import *
import sys
sys.path.append("../..")

from utils import *
import time


def evaluate(dataset_name, ed, fraction, mu, sigma, R, W):
    # Run BayesCP
    X, outliers_p = generator(dataset_name, fraction, mu, sigma)
    if ed == None:
        ed = X.shape[2]
    X = X[:, :, 0:ed]
    outliers_p = outliers_p[:, :, 0:ed]
    g = GraphTR(X=X, epsilon=np.sum(outliers_p), R=R, W=W, max_epoch=15, maxiters=15)
    model = g.run(outliers_p=outliers_p)
    return model


def eval_mu(dataset_name, R, W):
    TPRS = []
    FPRS = []
    mus = [0.01, 0.05, 0.1, 0.5, 1]
    # Run BayesCP
    for mu in mus:
        model = evaluate(dataset_name, 8000, 0.1, mu, 0.1, R, W)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'GraphTR/mu_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))

    with open(os.path.join(os.path.join('../../results', dataset_name), 'GraphTR/mu_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_sigma(dataset_name, R, W):
    TPRS = []
    FPRS = []
    # Run BayesCP
    sigmas = [0.01, 0.05, 0.1, 0.5, 1]
    for sigma in sigmas:
        model = evaluate(dataset_name, 8000, 0.1, 0, sigma, R, W)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'GraphTR/sigma_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))

    with open(os.path.join(os.path.join('../../results', dataset_name), 'GraphTR/sigma_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_ratio(dataset_name, R, W):
    TPRS = []
    FPRS = []
    # Run BayesCP
    for fraction in range(1, 11, 1):
        model = evaluate(dataset_name, 8000, fraction/100, 0, 0.1, R, W)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'GraphTR/ratio_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))

    with open(os.path.join(os.path.join('../../results', dataset_name), 'GraphTR/ratio_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


if __name__ == '__main__':
    start = time.time()
    eval_ratio('CERNET', 15, 2.5)
    end = time.time()
    print('entire ratio-CERNET consumption :', end - start)
    sys.stdout.flush()