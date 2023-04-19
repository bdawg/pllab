
import numpy as np
import time
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import timeit

setupstr= '''
def fntest():
    time.sleep(0.000001)
'''

def fntest():
    tm = 0.01
    t0 = time.perf_counter()
    while time.perf_counter()-t0 < tm:
        pass

setupstr= '''
def fntest():
    tm = 0.001
    t0 = time.perf_counter()
    while time.perf_counter()-t0 < tm:
        pass
'''


stmtstr= '''
fntest()
'''

nits = 100

t = timeit.timeit(setup=setupstr, stmt=stmtstr, number=nits)
runtime = t/nits * 1e3
print('Average time to run: %.3f ms' % runtime)