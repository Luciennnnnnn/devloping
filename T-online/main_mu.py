from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_mu('Abilene')
    end = time.time()
    print('entire mu-Abilene consumption :', end - start)