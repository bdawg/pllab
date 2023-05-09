import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')
plt.ion()

darkfiledir = '../pllab_data/darks/'
darkfile1 = None #'dark_psfcam_50us'
verbose = True


cam_syncdelay_ms = 12
extra_delay_ms  = 3
wait_time_ms = cam_syncdelay_ms + extra_delay_ms


# For PSFcam:
cnt1 = (257, 340)
wsz1 = 48
# For PLcam:
cnt2 = (160, 282)
wsz2 = 120

win1 = (cnt1[0]-wsz1//2, cnt1[0]+wsz1//2-1, cnt1[1]-wsz1//2, cnt1[1]+wsz1//2-1)
win2 = (cnt2[0]-wsz2//2, cnt2[0]+wsz2//2-1, cnt2[1]-wsz2//2, cnt2[1]+wsz2//2-1)

win=win2

if not 'cam1' in locals():
    cam1 = credcam(camera_index=0, darkfile=darkfile1, darkpath=darkfiledir, verbose=verbose)
    cam1.set_tint(0.00005, verbose=verbose)
    cam1.external_trigger(enabled=True, syncdelay=cam_syncdelay_ms, verbose=verbose)

print('Starting data acquisition')
startime = time.time()

cam1.reset_buffer()

for k in range(60): #120
    print(cam1.check_nims_buffer())
    time.sleep(0.25)

all_camims1 = cam1.get_buffer_images()
all_camims1 = all_camims1[:, win[0]:win[1], win[2]:win[3]]
imfluxes1 = np.sum(all_camims1, axis=(1, 2))

plt.figure(1)
plt.clf()
xlim = [0,20]
xlim = None
# plt.title('psfcam')
plt.plot(imfluxes1, '-x')
plt.xlim(xlim)
plt.ylabel('Window flux')



