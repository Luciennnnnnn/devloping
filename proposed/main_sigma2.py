from proposed import *
from cores import *


if __name__ == '__main__':
    start = time.time()
    eval_sigma('GEANT')
    end = time.time()
    print('Time Comsuption GEANT_sigma:', end - start)