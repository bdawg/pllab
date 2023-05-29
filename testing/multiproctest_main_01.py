import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import shared_memory

from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')
plt.ion()


"""
commsl is structures as follows:
[0]  - camera command send
[1]  - camera response
[2]  - cube_nims

[10]  - cam acquire. Set to 1 to acquire cube, and camprocess will set to 0 when acq complete.


"""

cam0_acquire = True
cam1_acquire = False

# Set up shared memory for both shareable list (for communication) and image data array
cam0_commsl_shmname = 'cam0_commsl'
cam0_imshm_shmname = 'cam0_imshm'
cam1_commsl_shmname = 'cam1_commsl'
cam1_imshm_shmname = 'cam1_imshm'

# To remove all shms if not unlinked:
try:
    shared_memory.SharedMemory(name=cam0_commsl_shmname).unlink()
    shared_memory.SharedMemory(name=cam0_imshm_shmname).unlink()
    shared_memory.SharedMemory(name=cam1_commsl_shmname).unlink()
    shared_memory.SharedMemory(name=cam1_imshm_shmname).unlink()
except:
    pass


#Define maximum memory allocated to shareable list and imdata
camdims = (640, 512) # Needs to be decided ahead of time
cube_nims = 1000
sl_nitems = 20
sl_maxitemlength = 100
cam0_commsl = shared_memory.ShareableList([' '*sl_maxitemlength]*sl_nitems, name=cam0_commsl_shmname)
cam1_commsl = shared_memory.ShareableList([' '*sl_maxitemlength]*sl_nitems, name=cam1_commsl_shmname)
# Once memory allocated, initialise with Nones
for k in range(sl_nitems):
    cam0_commsl[k] = None
    cam1_commsl[k] = None

init_cube = np.zeros((cube_nims, camdims[1], camdims[0]), dtype=np.int16)
cam0_imshm_obj = shared_memory.SharedMemory(name=cam0_imshm_shmname, create=True, size=init_cube.nbytes)
cam0_imshm = np.ndarray(init_cube.shape, dtype=init_cube.dtype, buffer=cam0_imshm_obj.buf)
cam0_imshm[:] = np.copy(init_cube)
cam1_imshm_obj = shared_memory.SharedMemory(name=cam1_imshm_shmname, create=True, size=init_cube.nbytes)
cam1_imshm = np.ndarray(init_cube.shape, dtype=init_cube.dtype, buffer=cam1_imshm_obj.buf)
cam1_imshm[:] = np.copy(init_cube)
del init_cube

# Add values to commsl
cam0_commsl[2] = cube_nims
cam1_commsl[2] = cube_nims
cam0_commsl[10] = 0
cam1_commsl[10] = 0

for k in range(20):
    print('Waiting %d sec...' % (20-k))
    time.sleep(1)

print('Acquiring cube of %d frames' % cube_nims)
if cam0_acquire:
    cam0_commsl[10] = 1
if cam1_acquire:
    cam1_commsl[10] = 1

while (cam0_commsl[10] + cam0_commsl[10]) > 0:
    print('Waiting for camprocesses to finish acquiring...')
    time.sleep(1)

print('Acqusition complete')

plt.figure(1)
cubefluxes0 = np.sum(cam0_imshm, axis=(1,2))
cubefluxes1 = np.sum(cam1_imshm, axis=(1,2))
plt.clf()
plt.subplot(211)
plt.plot(cubefluxes0)
plt.title('Camera 0 fluxes')
plt.subplot(212)
plt.plot(cubefluxes1)
plt.title('Camera 1 fluxes')

plt.figure(2)
dk = 0#cam0_imshm[0,:,:,]
for k in range(20):
    plt.clf()
    plt.imshow(cam0_imshm[k,:,:,]-dk, clim=[-200,5000])
    plt.colorbar()
    plt.title(k)
    plt.pause(0.5)





