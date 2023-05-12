import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import shared_memory

from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')
plt.ion()


# camnum = 0 # Logical camera number. Would be specified when launching script.
camdims = (640, 512) # Needs to be decided ahead of time

# Set up shared memory for both shareable list (for communication) and image data array
cam_commsl_shmname = 'cam0_commsl'
cam_imshm_shmname = 'cam0_imshm'

cam_commsl = shared_memory.ShareableList(name=cam_commsl_shmname)
cam_imshm_obj = shared_memory.SharedMemory(name=cam_imshm_shmname)
cube_nims = cam_commsl[2]
cam_imshm = np.ndarray((cube_nims, camdims[1], camdims[0]), dtype=np.int16, buffer=cam_imshm_obj.buf)



# Set up camera
darkfiledir = '../pllab_data/darks/'
darkfile1 = None #'dark_psfcam_50us'
verbose = True

cam_syncdelay_ms = 12
extra_delay_ms  = 3
wait_time_ms = cam_syncdelay_ms + extra_delay_ms

cam = credcam(camera_index=1, darkfile=darkfile1, darkpath=darkfiledir, verbose=verbose)
cam.external_trigger(enabled=True, syncdelay=cam_syncdelay_ms, verbose=verbose)
# cam.external_trigger(enabled=False)
# cam.send_command('set fps 300')

print('camprocess: Waiting for start command...')
# Wait for command to acquire a cube
while cam_commsl[10] == 0: # Block
    time.sleep(0.01)

# Acquire data
cam.reset_buffer()
startime = time.time()
print('camprocess: Starting acqusition')
while cam.check_nims_buffer() < cube_nims:
    print('camprocess: Acquired image %d of %d' % (cam.check_nims_buffer(), cube_nims))
    time.sleep(0.5)

# cube = np.copy(cam.get_buffer_images()[:cube_nims,:,:])
cam_imshm[:] = np.copy(cam.get_buffer_images()[:cube_nims,:,:])
cam_commsl[10] = 0
print('camprocess: Acquisition complete')

# print('camprocess: Waiting 10 sec...')
# time.sleep(10)
# print('camprocess: Terminating process')

# plt.figure(2)
# for k in range(20):
#     plt.clf()
#     plt.imshow(cube[k,:,:,], clim=[-200,5000])
#     plt.colorbar()
#     plt.title(k)
#     plt.pause(0.5)

# cubefluxes0 = np.sum(cube, axis=(1,2))
# plt.clf()
# plt.plot(cubefluxes0)

