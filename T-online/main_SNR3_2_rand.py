from T_online import *
from cores2 import *


if __name__ == '__main__':
    start = time.time()
    eval_SNR('CERNET', init="random")
    end = time.time()
    print('Time Comsuption CERNET_SNR:', end - start)