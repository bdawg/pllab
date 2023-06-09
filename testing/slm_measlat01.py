import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import *
from plcams import *
import matplotlib
matplotlib.use('TkAgg')


# lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'

darkfile = 'dark_psfcam_50us'

slm = plslm(lutfile=lutfile)
cam = credcam(camera_index=1, darkfile=darkfile, verbose=True)

# Make array of SLM images to cycle through
all_slmimages = []
# image0 = np.ones((slm.slmdims[0], slm.slmdims[1])) * 128
image0 = np.zeros((slm.slmdims[0], slm.slmdims[1]))
image0 = image0.round().astype('uint8')
all_slmimages.append(image0)

# slm.makeramp(xslope=0.1, showplot=True, sendtoslm=True)
slm.makestripes(period=50, ampl=60, phi=0, type='square', showplot=True, sendtoslm=False)
image1 = slm.nextim
image1 = image1.round().astype('uint8')
all_slmimages.append(image1)
all_slmimages = np.array(all_slmimages)


# cam.take_dark(darkfile='dark_psfcam_50us', save=True)


nloops = 10
sleeptime = 0.001
n_imstolog = 1

slm.slmwrite(all_slmimages[0, :, :], showplot=False)
time.sleep(0.5)

cam.set_nims_tolog(n_imstolog)
all_camims = np.zeros((nloops*len(all_slmimages), cam.camdims[1], cam.camdims[0]), dtype=np.int16)
all_imtimes = np.zeros(nloops*len(all_slmimages))
count = 0
for k in range(nloops):
    for l in range(len(all_slmimages)):
        # slmim = all_slmimages[l, :, :]
        # slm.slmwrite(slmim, showplot=False)
        # time.sleep(sleeptime)

        cam.goodtimer(1)

        # camim = cam.get_n_images(return_ims=True, coadd=False, subtract_dark=False)
        camim = cam.get_latest_image(waitfornewframe=False, return_im=False)
        # all_camims[count, :, :] = camim
        all_imtimes[count] = cam.loggedims_times_arr[0]
        count += 1

print('Done')


cnt = (257, 340)
wsz = 48 #96

win = (cnt[0]-wsz//2, cnt[0]+wsz//2-1, cnt[1]-wsz//2, cnt[1]+wsz//2-1)

all_croppedims = all_camims[:,win[0]:win[1], win[2]:win[3]]
imfluxes = np.sum(all_croppedims, axis=(1,2))
eltime = all_imtimes- all_imtimes[0]
plt.clf()
plt.subplot(2,1,1)
plt.plot(imfluxes, '-x')
plt.ylabel('Window flux')
plt.subplot(2,1,2)
# plt.plot(eltime,'-x')
# plt.ylabel('Frame time from start (s)')
plt.plot(np.diff(all_imtimes*1000),'-x')
plt.ylabel('Time between frames (ms)')
plt.tight_layout()


# offset = 0
# for k in range(10): #all_camims.shape[0]):
#     plt.clf()
#     plt.imshow(all_croppedims[k,:,:])
#     plt.title(k+offset)
#     plt.pause(0.5)

