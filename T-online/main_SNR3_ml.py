from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_SNR('CERNET', init="svd")
    end = time.time()
    print('Time Comsuption CERNET_SNR:', end - start)