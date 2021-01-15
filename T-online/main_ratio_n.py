from T_online import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_ratio2('Abilene')
    end = time.time()
    print('entire ratio-Abilene consumption :', end - start)