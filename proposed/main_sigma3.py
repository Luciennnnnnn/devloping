from proposed import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_sigma('CERNET')
    end = time.time()
    print(end - start)
    print('Time Comsuption CERNET_sigma:', end - start)