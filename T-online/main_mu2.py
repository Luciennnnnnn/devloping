from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_mu('GEANT')
    end = time.time()
    print('entire mu-GEANT consumption :', end - start)