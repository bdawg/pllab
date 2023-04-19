import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import *
from plcams import *
import matplotlib
matplotlib.use('TkAgg')

"""
Use variable wait time instead of callbacks, since those 
dont seem to work properly with multiple cameras...
"""

lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'
darkfile1 = 'dark_psfcam_50us'

slm = plslm(lutfile=lutfile)
cam = credcam(camera_index=1, darkfile=darkfile1, verbose=True)

# Make array of SLM images to cycle through
all_slmimages = []
image0 = np.zeros((slm.slmdims[0], slm.slmdims[1]))
image0 = image0.round().astype('uint8')
all_slmimages.append(image0)
slm.makestripes(period=50, ampl=60, phi=0, type='square', showplot=True, sendtoslm=False)
image1 = slm.nextim
image1 = image1.round().astype('uint8')
all_slmimages.append(image1)
all_slmimages = np.array(all_slmimages)


cnt = (257, 340)
wsz = 48 #96

nloops = 100
# n_imstolog = 1

save_all_ims = True

slm.slmwrite(all_slmimages[0, :, :], showplot=False)
time.sleep(0.5)
# cam.set_nims_tolog(n_imstolog)

wait_time_ms = 10

# wait_times = np.arange(0, 10.1, 0.1)
# wait_times = np.arange(0, 20.2, 0.2)
# wait_times = np.arange(0, 4, 1)
wait_times = np.arange(10.0, 20.2, 0.2)



all_imfluxes = []
all_eltimes = []
all_ims = []

for wait_time_ms in wait_times:
    print('Doing wait time %f ms' % wait_time_ms)

    all_camims = np.zeros((nloops*len(all_slmimages), cam.camdims[1], cam.camdims[0]), dtype=np.int16)
    all_imtimes = np.zeros(nloops*len(all_slmimages))
    count = 0
    for k in range(nloops):
        for l in range(len(all_slmimages)):
            slmim = all_slmimages[l, :, :]
            slm.slmwrite(slmim, showplot=False)

            cam.goodtimer(wait_time_ms)

            camim = cam.get_latest_image(waitfornewframe=False, return_im=True)
            all_camims[count, :, :] = camim
            all_imtimes[count] = cam.loggedims_times_arr[0]
            count += 1
    print('Done')

    all_camims = all_camims - cam.dark
    win = (cnt[0]-wsz//2, cnt[0]+wsz//2-1, cnt[1]-wsz//2, cnt[1]+wsz//2-1)
    all_croppedims = all_camims[:,win[0]:win[1], win[2]:win[3]]
    imfluxes = np.sum(all_croppedims, axis=(1,2))
    eltime = all_imtimes- all_imtimes[0]

    all_imfluxes.append(imfluxes)
    all_eltimes.append(eltime)
    if save_all_ims:
        all_ims.append(all_croppedims)

all_imfluxes = np.asarray(all_imfluxes)
all_eltimes = np.asarray(all_eltimes)

np.savez('timingdata.npz', all_imfluxes=all_imfluxes, all_eltimes=all_eltimes, wait_times=wait_times)
if save_all_ims:
    all_ims = np.asarray(all_ims)
    np.savez('timingdata_images.npz', all_ims=all_ims, all_imfluxes=all_imfluxes, all_eltimes=all_eltimes,
             wait_times=wait_times)


plt.plot(np.diff(all_eltimes[0,:]),'-+')


cmap = 'gray'
plt.clf()
plt.subplot(2,1,1)
plt.imshow(all_imfluxes, cmap=cmap, aspect='auto')
plt.colorbar()
plt.xlabel('Frame number')
plt.ylabel('Wait time (index)')
plt.title('Flux')
plt.subplot(2,1,2)
plt.imshow(all_eltimes*1e3, cmap=cmap, aspect='auto')
plt.xlabel('Frame number')
plt.ylabel('Wait time (index)')
plt.title('Elapsed times')
plt.colorbar()
plt.tight_layout()



# plt.clf()
# plt.subplot(2,1,1)
# plt.plot(imfluxes, '-x')
# plt.ylabel('Window flux')
# plt.subplot(2,1,2)
# # plt.plot(eltime,'-x')
# # plt.ylabel('Frame time from start (s)')
# plt.plot(np.diff(all_imtimes*1000),'-x')
# plt.ylabel('Time between frames (ms)')
# plt.tight_layout()
