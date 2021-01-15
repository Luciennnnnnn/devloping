from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_time('GEANT', 8000)
    end = time.time()
    print('Time Comsuption GEANT_time:', end - start)