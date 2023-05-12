import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import shared_memory

from plslm import plslm
from plcams import credcam
import matplotlib
import subprocess
matplotlib.use('TkAgg')
plt.ion()

def goodtimer(time_ms):
    tm = time_ms / 1000
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < tm:
        pass

"""
commsl is structures as follows:
[0]  - camera command send
[1]  - camera response
[2]  - cube_nims
[3]  - camera dim 0
[4]  - camera dim 1
[5]  - camera process ready
[10] - cam acquire. Set to 1 to acquire cube, and camprocess will set to 0 when acq complete.


"""
camids = {'pl': 'c86ca65',
          'psf': '1de3043',
          'refl': '1de3be41'}

cam0_id = camids['psf']
cam1_id = camids['pl']
cam0_acquire = True
cam1_acquire = True

# Set up shared memory for both shareable list (for communication) and image data array
cam0_commsl_shmname = 'cam0_commsl'
cam0_imshm_shmname = 'cam0_imshm'
cam1_commsl_shmname = 'cam1_commsl'
cam1_imshm_shmname = 'cam1_imshm'

# Note - ultradict could be used to have a shred dictionary, rather than list -
# https://github.com/ronny-rentner/UltraDict

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
cube_nims = 100
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
cam0_commsl[3] = camdims[0]
cam1_commsl[3] = camdims[0]
cam0_commsl[4] = camdims[1]
cam1_commsl[4] = camdims[1]
cam0_commsl[5] = 0
cam1_commsl[5] = 0
cam0_commsl[10] = 0
cam1_commsl[10] = 0


# Set up camera processes
verbose_state = '1'
if cam0_acquire:
    args0 = ['python', 'plcam_camprocess.py', cam0_id, cam0_commsl_shmname, cam0_imshm_shmname, verbose_state]
    subproc_camproc0 = subprocess.Popen(args0)
    while cam0_commsl[5] != 1:
        print('Waiting for cam0 setup to complete...')
        time.sleep(5)
    print('cam0 ready')
if cam1_acquire:
    args1 = ['python', 'plcam_camprocess.py', cam1_id, cam1_commsl_shmname, cam1_imshm_shmname, verbose_state]
    subproc_camproc1 = subprocess.Popen(args1)
    while cam1_commsl[5] != 1:
        print('Waiting for cam1 setup to complete...')
        time.sleep(5)
    print('cam1 ready')


# Set up SLM
datadir = '../pllab_data/'
# slmims_filename = 'slmcube_varyingstripes_0-60_10frm_01.npz'
slmims_filename = 'slmcube_alternatingstripes_0-60_10frm_01.npz'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'
nloops = 10

cam_syncdelay_ms = 12
extra_delay_ms  = 3
wait_time_ms = cam_syncdelay_ms + extra_delay_ms

slm = plslm(lutfile=lutfile)
slmimdataf = np.load(datadir + slmims_filename)
slmims = slmimdataf['all_slmims']
if type(slmims[0, 0, 0]) is not np.int8:
    print('Error: input SLM cube not int8')
all_slmim_params = slmimdataf['all_slmim_params']
n_slmims = slmims.shape[0]
total_nims = nloops * n_slmims


print('Acquiring cube of %d frames' % cube_nims)
if cam0_acquire:
    cam0_commsl[10] = 1
if cam1_acquire:
    cam1_commsl[10] = 1

time.sleep(0.1) # Hack, to allow for polling rate of camprocess to cam_commsl[10]

startime = time.time()
count=0
for k in range(nloops):
    for l in range(n_slmims):
        slmim = slmims[l, :, :]
        slm.slmwrite(slmim, showplot=False, skip_readycheck=True)
        count += 1

        goodtimer(wait_time_ms)


while (cam0_commsl[10] + cam1_commsl[10]) > 0:
    print('Waiting for camprocesses to finish acquiring...')
    time.sleep(1)

print('Done - elapsed time %.2f seconds' % (time.time() - startime))
print('Acqusition complete')



# For PSFcam:
cnt1 = (257, 340)
wsz1 = 48
# For PLcam:
cnt2 = (160, 282)
wsz2 = 120

win1 = (cnt1[0]-wsz1//2, cnt1[0]+wsz1//2-1, cnt1[1]-wsz1//2, cnt1[1]+wsz1//2-1)
win2 = (cnt2[0]-wsz2//2, cnt2[0]+wsz2//2-1, cnt2[1]-wsz2//2, cnt2[1]+wsz2//2-1)

cam0_croppedcube = cam0_imshm[:, win1[0]:win1[1], win1[2]:win1[3]]
cam1_croppedcube = cam1_imshm[:, win2[0]:win2[1], win2[2]:win2[3]]

plt.figure(1)
cubefluxes0 = np.sum(cam0_croppedcube, axis=(1,2))
cubefluxes1 = np.sum(cam1_croppedcube, axis=(1,2))
plt.clf()
plt.subplot(211)
plt.plot(cubefluxes0)
plt.title('Camera 0 fluxes')
plt.subplot(212)
plt.plot(cubefluxes1)
plt.title('Camera 1 fluxes')

plt.figure(2)
dk = 0#cam0_imshm[0,:,:,]
for k in range(10):
    plt.clf()
    plt.imshow(cam0_croppedcube[k,:,:,]-dk, clim=[-200,5000])
    plt.colorbar()
    plt.title(k)
    plt.pause(0.5)





