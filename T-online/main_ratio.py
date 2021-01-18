from T_online import *
from cores import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VITDA')
    parser.add_argument('--noise_scheme', type=str, default=None)
    parser.add_argument('--outliers_scheme', type=str, default='Gaussian')

    args = parser.parse_args()

    parameters = {"ed": 8000, "R":6, "mu":0, "sigma":0.1, "SNR": None, 
                "noise_scheme": args.noise_scheme, 'outliers_scheme': args.outliers_scheme, 'init': 'svd'}

    start = time.time()
    eval_ratio('Abilene', parameters)
    end = time.time()
    logging.info('Time consumption in Abilene on ratio: %f' %(end - start)/60)