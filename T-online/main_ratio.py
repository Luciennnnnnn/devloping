import logging
from T_online import *
from cores import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T-online')
    parser.add_argument('--dataset', type=str, default='Abilene')
    parser.add_argument('--noise_scheme', type=str, default=None)
    parser.add_argument('--outliers_scheme', type=str, default='Gaussian')
    parser.add_argument('--theta', type=float, default=0.1)

    args = parser.parse_args()
    ed = 8000
    R = 6
    mu = 0
    sigma = 0.1
    SNR = None
    init_scheme = 'svd'
    parameters = {"ed": 8000, "R":6, "mu":0, "sigma":0.1, "SNR": SNR, "missing_ratio": 0,
                "noise_scheme": args.noise_scheme, 'outliers_scheme': args.outliers_scheme, 'init': init_scheme}
    logging.info('T-online ratio---- dataset: %s, ed: %d, R: %d, mu: %d, sigma: %f, noise_scheme: %s, outliers_scheme: %s, init: %s'%(args.dataset, ed, R, mu, sigma, args.noise_scheme, args.outliers_scheme, init_scheme))

    start = time.time()
    eval_ratio(args.dataset, parameters)
    end = time.time()
    logging.info('Time consumption in %s on ratio: %f' %(args.dataset, (end - start)/60))