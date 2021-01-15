from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_sigma('GEANT')
    end = time.time()
    print('entire sigma-GEANT consumption :', end - start)