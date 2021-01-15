from proposed3 import *
from cores3 import *


if __name__ == '__main__':
    start = time.time()
    eval_sigma('CERNET', init="ml")
    end = time.time()
    print(end - start)
    print('Time Comsuption CERNET_sigma:', end - start)