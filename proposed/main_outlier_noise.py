from proposed import *
from cores import *

if __name__ == '__main__':
    eval_mu('Abilene', noise_scheme='outlier', file_prefix='outlier_noise')
    eval_sigma('Abilene')