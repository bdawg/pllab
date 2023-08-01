from multiprocessing import Pool
import time
import numpy as np

def f(x):
    print('Running f function')
    for k in range(10):
        print('%d' % k)
        print(x)
        print(' ')
        time.sleep(0.1)
    # return x*x

if __name__ == '__main__':
    singlespeck = True
    cycles_per_pupil_range = [0, 2.5]
    ampl1_range = [0.1, 1]
    rot_range = [-180, 180]
    phi_range = [0, 100]

    # p = [cycles_per_pupil_range, ampl1_range, rot_range, phi_range]
    # pa = np.array(p)
    # p = {}
    # p['cycles_per_pupil_range'] = cycles_per_pupil_range
    # p['ampl1_range'] = ampl1_range
    # p['rot_range'] = rot_range
    # p['phi_range'] = phi_range

    transformed_list = list(zip(*p))

    # xs = [np.array((1,2,3)), np.array((1,2,3))*2, np.array((1,2,3))*4]
    with Pool(5) as p:
        print(p.map(f, transformed_list))


