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


multiprocessing.set_start_method('fork')

shm1 = multiprocessing.shared_memory.Sh

c = np.arange(1,5,0.2)
print(c)
proc = Process(target=myfn, args=(2, 'hi', c))
proc.start()





