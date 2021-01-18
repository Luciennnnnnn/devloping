from proposed import *
from cores import *
import argparse


if __name__ == '__main__':
    # run experiment for metrics outliers ratio
    
    parser = argparse.ArgumentParser(description='dd')
    parser.add_argument('--noise_scheme', type=str, default=None)
    parser.add_argument('--outliers_scheme', type=str, default='Gaussian')

    args = parser.parse_args()
    if args.noise_scheme == None:
        print('None')
        
    parameters = {"ed": 8000, "R":6, "K":6, "mu":0, "sigma":0.1, 
                "SNR":None, "noise_scheme": args.None, 'outliers_scheme': args.outliers_scheme}
    start = time.time()
    eval_ratio('Abilene', parameters)
    end = time.time()
    print('Time consumption in Abilene on ratio:', end - start)

    start = time.time()
    eval_ratio('GEANT', parameters)
    end = time.time()
    print('Time consumption in GEANT on ratio:', end - start)

    start = time.time()
    eval_ratio('CERNET', parameters)
    end = time.time()
    print('Time consumption in CERNET on ratio:', end - start)