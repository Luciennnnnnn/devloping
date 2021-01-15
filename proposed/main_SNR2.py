from proposed import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_SNR('GEANT')
    end = time.time()
    print('Time Comsuption GEANT_SNR:', end - start)