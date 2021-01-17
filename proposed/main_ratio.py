from proposed import *
from cores import *

if __name__ == '__main__':
    # run experiment for metrics outliers ratio
    parameters = {"ed": 8000, "R":6, "K":6, "mu":0, "sigma":0.1, 
                "SNR":None, "noise_scheme": None, 'outliers_scheme': 'Gaussian'}
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