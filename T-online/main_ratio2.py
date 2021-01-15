from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_ratio('GEANT')
    end = time.time()
    print('entire ratio-GEANT consumption :', end - start)