import logging
from proposed import *
from cores import *
import argparse


if __name__ == '__main__':
    # run experiment for metrics outliers ratio
    
    parser = argparse.ArgumentParser(description='T-online')
    parser.add_argument('--noise_scheme', type=str, default=None)
    parser.add_argument('--outliers_scheme', type=str, default=None)

    args = parser.parse_args()

    ed = 8000
    R = 2
    K = 6
    mu = 0
    sigma = 0.1
    theta = 0.1
    SNR = None
    parameters = {"ed": ed, "R":R, "K":K, "mu":mu, "sigma":sigma, "theta": theta, "fraction": 0, 
                "SNR":SNR, "noise_scheme": args.noise_scheme, 'outliers_scheme': args.outliers_scheme}
    
    logging.info('recover --- ed: %d, R: %d, K: %d, mu: %d, sigma: %f, theta: %f, noise_scheme: %s, outliers_scheme: %s'%(ed, R, K, mu, sigma, theta, args.noise_scheme, args.outliers_scheme))     
    start = time.time()
    eval_missing_ratio('Abilene', parameters)
    end = time.time()
    logging.info('Time consumption in Abilene on ratio: %f' %((end - start)/60))

    start = time.time()
    eval_missing_ratio('GEANT', parameters)
    end = time.time()
    logging.info('Time consumption in GEANT on ratio: %f' %((end - start)/60))

    start = time.time()
    eval_missing_ratio('CERNET', parameters)
    end = time.time()
    logging.info('Time consumption in CERNET on ratio: %f' %((end - start)/60))