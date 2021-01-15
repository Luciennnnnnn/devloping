from T_online import *
from cores import *

if __name__ == '__main__':
    start = time.time()
    eval_mu('VData50', 672)
    end = time.time()
    print(end - start)