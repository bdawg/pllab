import time
from multiprocessing import Process
import multiprocessing
import numpy as np

def myfn(a, b, c):
    print('Started function in other process, got value:')
    print(b)
    time.sleep(a)
    print(np.sqrt(c))
    print('Ending.')
    return 23

# def myfn2():
#     a = np.sqrt(np.random.rand(1000,1000,300))

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    c = np.arange(1,5,0.2)
    print(c)
    proc = Process(target=myfn, args=(2, 'hi', c))
    proc.start()





