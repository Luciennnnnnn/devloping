from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_sigma('CERNET')
    end = time.time()
    print('entire sigma-CERNET consumption :', end - start)