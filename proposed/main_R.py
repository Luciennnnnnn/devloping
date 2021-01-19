from proposed import *
from cores import *
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='T-online')
    parser.add_argument('--noise_scheme', type=str, default=None)
    parser.add_argument('--outliers_scheme', type=str, default='Gaussian')

    args = parser.parse_args()

    ed = 8000
    R = 6
    K = 6
    mu = 0
    sigma = 0.1
    SNR = None
    parameters = {"ed": ed, "R":R, "K":K, "mu":mu, "sigma":sigma, "fraction":0.1,
                "SNR":SNR, "noise_scheme": args.noise_scheme, 'outliers_scheme': args.outliers_scheme}
    
    logging.info('ed: %d, R: %d, K: %d, mu: %d, sigma: %f, noise_scheme: %s, outliers_scheme: %s'%(ed, R, K, mu, sigma, args.noise_scheme, args.outliers_scheme))     
    start = time.time()
    eval_R('Abilene', parameters)
    end = time.time()
    logging.info('Time consumption in Abilene on ratio: %f' %((end - start)/60))

    start = time.time()
    eval_R('GEANT', parameters)
    end = time.time()
    logging.info('Time consumption in GEANT on ratio: %f' %((end - start)/60))

    start = time.time()
    eval_R('CERNET', parameters)
    end = time.time()
    logging.info('Time consumption in CERNET on ratio: %f' %((end - start)/60))