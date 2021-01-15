from proposed import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_time('CERNET', 8000)
    end = time.time()
    print('Time Comsuption CERNET_time:', end - start)