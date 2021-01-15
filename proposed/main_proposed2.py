from proposed import *


def evaluate(dataset_name, ed, fraction, mu, sigma, R):
    # Run BayesCP
    Y, outliers_p = generator(dataset_name, fraction, mu, sigma)
    if ed == None:
        ed = Y.shape[2]
    model = BCPF_IC(Y=Y, outliers_p=outliers_p, start=0, end=ed, maxRank=R, maxiters=30, tol=1e-4, verbose=False)
    return model


def eval_time(dataset_name, ed):
    TPRS = [[] for _ in range(ed)]
    FPRS = [[] for _ in range(ed)]
    RSE = [[] for _ in range(ed)]
    for i in range(10):
        # Run BayesCP
        model = evaluate(dataset_name, ed, 0.1, 0, 0.1, 15)
        for j in range(len(model['TPRS'])):
            TPRS[j].append(model['TPRS'][j])
            FPRS[j].append(model['FPRS'][j])
            RSE[j].append(model['RSE'][j])

    T_means = []
    T_stderr = []
    F_means = []
    F_stderr = []
    RSE_means = []
    RSE_stderr = []

    for i in range(len(TPRS)):
        T_means.append(np.mean(TPRS[i]))
        T_stderr.append(np.std(TPRS[i]) / sqrt(10))
    for i in range(len(FPRS)):
        F_means.append(np.mean(FPRS[i]))
        F_stderr.append(np.std(FPRS[i]) / sqrt(10))
    for i in range(len(RSE)):
        RSE_means.append(np.mean(RSE[i]))
        RSE_stderr.append(np.std(RSE[i]) / sqrt(10))

    with open(os.path.join(os.path.join('results', dataset_name), 'TPRS_mean.json'), 'w') as FD:
        FD.write(json.dumps(T_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'TPRS_stderr.json'), 'w') as FD:
        FD.write(json.dumps(T_stderr))

    with open(os.path.join(os.path.join('results', dataset_name), 'FPRS_mean.json'), 'w') as FD:
        FD.write(json.dumps(F_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'FPRS_stderr.json'), 'w') as FD:
        FD.write(json.dumps(F_stderr))

    with open(os.path.join(os.path.join('results', dataset_name), 'RSE_mean.json'), 'w') as FD:
        FD.write(json.dumps(F_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'RSE_stderr.json'), 'w') as FD:
        FD.write(json.dumps(F_stderr))


def eval_R(dataset_name):
    TPRS = [[] for _ in range(5, 38, 3)]
    FPRS = [[] for _ in range(5, 38, 3)]
    for i in range(10):
        # Run BayesCP
        print(i)
        j = 0
        start = time.time()
        for R in range(5, 38, 3):
            model = evaluate(dataset_name, None, 0.1, 0, 0.1, R)
            TPRS[j].append(model['precision'])
            FPRS[j].append(model['FPR'])
            j += 1
        end = time.time()
        print(end - start)

    T_means = []
    T_stderr = []
    F_means = []
    F_stderr = []

    for i in range(len(TPRS)):
        T_means.append(np.mean(TPRS[i]))
        T_stderr.append(np.std(TPRS[i]) / sqrt(10))
    for i in range(len(FPRS)):
        F_means.append(np.mean(FPRS[i]))
        F_stderr.append(np.std(FPRS[i]) / sqrt(10))

    with open(os.path.join(os.path.join('results', dataset_name), 'R_TPRS_mean.json'), 'w') as FD:
        FD.write(json.dumps(T_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'R_TPRS_stderr.json'), 'w') as FD:
        FD.write(json.dumps(T_stderr))

    with open(os.path.join(os.path.join('results', dataset_name), 'R_FPRS_mean.json'), 'w') as FD:
        FD.write(json.dumps(F_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'R_FPRS_stderr.json'), 'w') as FD:
        FD.write(json.dumps(F_stderr))


def eval_mu(dataset_name):
    TPRS = [[] for _ in range(1, 11, 1)]
    FPRS = [[] for _ in range(1, 11, 1)]
    for i in range(10):
        # Run BayesCP
        j = 0
        for mu in range(1, 11, 1):
            model = evaluate(dataset_name, None, 0.1, mu/10, 0.1, 15)
            TPRS[j].append(model['precision'])
            FPRS[j].append(model['FPR'])
            j += 1

    T_means = []
    T_stderr = []
    F_means = []
    F_stderr = []

    for i in range(len(TPRS)):
        T_means.append(np.mean(TPRS[i]))
        T_stderr.append(np.std(TPRS[i]) / sqrt(10))
    for i in range(len(FPRS)):
        F_means.append(np.mean(FPRS[i]))
        F_stderr.append(np.std(FPRS[i]) / sqrt(10))

    with open(os.path.join(os.path.join('results', dataset_name), 'mu_TPRS_mean.json'), 'w') as FD:
        FD.write(json.dumps(T_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'mu_TPRS_stderr.json'), 'w') as FD:
        FD.write(json.dumps(T_stderr))

    with open(os.path.join(os.path.join('results', dataset_name), 'mu_FPRS_mean.json'), 'w') as FD:
        FD.write(json.dumps(F_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'mu_FPRS_stderr.json'), 'w') as FD:
        FD.write(json.dumps(F_stderr))


def eval_sigma(dataset_name):
    TPRS = [[] for _ in range(1, 11, 1)]
    FPRS = [[] for _ in range(1, 11, 1)]
    for i in range(10):
        # Run BayesCP
        j = 0
        for sigma in range(1, 11, 1):
            model = evaluate(dataset_name, None, 0.1, 0, sigma/10, 15)
            TPRS[j].append(model['precision'])
            FPRS[j].append(model['FPR'])
            j += 1

    T_means = []
    T_stderr = []
    F_means = []
    F_stderr = []

    for i in range(len(TPRS)):
        T_means.append(np.mean(TPRS[i]))
        T_stderr.append(np.std(TPRS[i]) / sqrt(10))
    for i in range(len(FPRS)):
        F_means.append(np.mean(FPRS[i]))
        F_stderr.append(np.std(FPRS[i]) / sqrt(10))

    with open(os.path.join(os.path.join('results', dataset_name), 'sigma_TPRS_mean.json'), 'w') as FD:
        FD.write(json.dumps(T_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'sigma_TPRS_stderr.json'), 'w') as FD:
        FD.write(json.dumps(T_stderr))

    with open(os.path.join(os.path.join('results', dataset_name), 'sigma_FPRS_mean.json'), 'w') as FD:
        FD.write(json.dumps(F_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'sigma_FPRS_stderr.json'), 'w') as FD:
        FD.write(json.dumps(F_stderr))


def eval_ratio(dataset_name):
    TPRS = [[] for _ in range(1, 11, 1)]
    FPRS = [[] for _ in range(1, 11, 1)]
    for i in range(10):
        # Run BayesCP
        j = 0
        for fraction in range(1, 11, 1):
            model = evaluate(dataset_name, None, fraction / 100, 0, 0.1, 15)
            TPRS[j].append(model['precision'])
            FPRS[j].append(model['FPR'])
            j += 1

    T_means = []
    T_stderr = []
    F_means = []
    F_stderr = []

    for i in range(len(TPRS)):
        T_means.append(np.mean(TPRS[i]))
        T_stderr.append(np.std(TPRS[i]) / sqrt(10))
    for i in range(len(FPRS)):
        F_means.append(np.mean(FPRS[i]))
        F_stderr.append(np.std(FPRS[i]) / sqrt(10))

    with open(os.path.join(os.path.join('results', dataset_name), 'ratio_TPRS_mean.json'), 'w') as FD:
        FD.write(json.dumps(T_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'ratio_TPRS_stderr.json'), 'w') as FD:
        FD.write(json.dumps(T_stderr))

    with open(os.path.join(os.path.join('results', dataset_name), 'ratio_FPRS_mean.json'), 'w') as FD:
        FD.write(json.dumps(F_means))

    with open(os.path.join(os.path.join('results', dataset_name), 'ratio_FPRS_stderr.json'), 'w') as FD:
        FD.write(json.dumps(F_stderr))


if __name__ == '__main__':
    start = time.time()
    eval_mu('Abilene')
    end = time.time()
    print(end - start)
    #
    # start = time.time()
    # eval_mu('Abilene')
    # end = time.time()
    # print(end - start)
    #
    # start = time.time()
    # eval_sigma('Abilene')
    # end = time.time()
    # print(end - start)
    #
    # start = time.time()
    # eval_ratio('Abilene')
    # end = time.time()
    # print(end - start)
    #eval_R('GEANT')
    #eval_R('CERNET')