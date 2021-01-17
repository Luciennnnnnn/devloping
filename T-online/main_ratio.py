from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    
    parameters = {"ed": 8000, "R":6, "mu":0, "sigma":0.1, "SNR": None, 
                "noise_scheme": 'outlier', 'outliers_scheme': 'Gaussian', 'init': 'svd'}

    eval_ratio('Abilene', parameters)
    end = time.time()
    print('entire ratio-Abilene consumption :', end - start)