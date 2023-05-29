import numpy as np
import time
from multiprocessing import shared_memory
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()


def run_realtime_display(shm_objs, slmdims=(1024,1024)):
    if shm_objs[0] is not None:
        sl = shared_memory.ShareableList(name=shm_objs[0])
        cube_nims = sl[2]
        dims = [sl[3], sl[4]]
        cam_imshm0 = np.ndarray((cube_nims, dims[1], dims[0]), dtype=np.int16,
                               buffer=shm_objs[1].buf)
    else:
        cam_imshm0 = np.zeros((1,1))

    if shm_objs[2] is not None:
        sl = shared_memory.ShareableList(name=shm_objs[2])
        dims = [sl[3], sl[4]]
        cam_imshm1 = np.ndarray((cube_nims, dims[1], dims[0]), dtype=np.int16,
                               buffer=shm_objs[3].buf)
    else:
        cam_imshm1 = np.zeros((1, 1))

    if shm_objs[4] is not None:
        sl = shared_memory.ShareableList(name=shm_objs[4])
        dims = [sl[3], sl[4]]
        cam_imshm2 = np.ndarray((cube_nims, dims[1], dims[0]),  dtype=np.int16,
                               buffer=shm_objs[5].buf)
    else:
        cam_imshm2 = np.zeros((1, 1))

    if shm_objs[6] is not None:
        dims = slmdims
        slm_imshm = np.ndarray((cube_nims, dims[1], dims[0]), dtype=np.int8,
                                buffer=shm_objs[6].buf)
    else:
        slm_imshm = np.zeros((1, 1))

    plt.figure(1)
    plt.clf()

    while True:
        plt.subplot(2,2,1)
        plt.imshow(cam_imshm0[0,:,:])
        plt.subplot(2,2,2)
        plt.imshow(cam_imshm1[0,:,:])
        plt.subplot(2,2,3)
        plt.imshow(cam_imshm2[0,:,:])
        plt.subplot(2,2,4)
        plt.imshow(slm_imshm[0,:,:])

        plt.pause(0.05)





"""
Command line arguments (or 'None' if not required)
cam0_commsl_shmname, cam0_imshm_shmname, cam1_commsl_shmname, cam1_imshm_shmname, 
cam2_commsl_shmname, cam2_imshm_shmname, slm_shmname
"""
if __name__ == '__main__':
    args = sys.argv
    shm_objs = []
    if len(args) == 1:
        args = ['', 'cam0_commsl', 'cam0_imshm', 'cam1_commsl', 'cam1_imshm',
                'None', 'None', 'None']
    for k in range(1,len(args)):
        if args[k] == 'None':
            shm_objs.append(None)
        else:
            shm_objs.append(shared_memory.SharedMemory(name=args[k]))

    run_realtime_display(shm_objs)
