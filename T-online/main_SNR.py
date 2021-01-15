from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_SNR('Abilene')
    end = time.time()
    print('Time Comsuption Abilene_SNR:', end - start)