from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_mu('CERNET')
    end = time.time()
    print('entire mu-CERNET consumption :', end - start)