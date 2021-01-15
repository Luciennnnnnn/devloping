from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_sigma('Abilene')
    end = time.time()
    print('entire sigma-Abilene consumption :', end - start)