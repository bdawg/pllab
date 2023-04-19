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


nloops = 4
n_imstolog = 300

slm.slmwrite(all_slmimages[0, :, :], showplot=False)
time.sleep(0.5)

cam.set_nims_tolog(n_imstolog)
all_camims = np.zeros((n_imstolog, nloops*len(all_slmimages), cam.camdims[1], cam.camdims[0]), dtype=np.int16)
all_imtimes = np.zeros((n_imstolog, nloops*len(all_slmimages)))
all_startimes = np.zeros(nloops*len(all_slmimages))

count = 0
for k in range(nloops):
    for l in range(len(all_slmimages)):
        slmim = all_slmimages[l, :, :]
        slm.slmwrite(slmim, showplot=False)

        all_startimes[count] = time.perf_counter()

        # camim = cam.get_n_images(return_ims=True, coadd=False, subtract_dark=False)
        ## camim = cam.get_latest_image(waitfornewframe=True, return_im=True)
        # Test non-blocking
        cam.get_n_images(blocking=False)
        while cam.nims_lefttolog > 0:
            time.sleep(0.1)
        camim = cam.loggedims_cube
        print('Got all images.')

        camim = camim - cam.dark
        all_camims[:, count, :, :] = camim
        all_imtimes[:, count] = cam.loggedims_times_arr
        count += 1

print('Done')


cnt = (257, 340)
wsz = 96

win = (cnt[0]-wsz//2, cnt[0]+wsz//2-1, cnt[1]-wsz//2, cnt[1]+wsz//2-1)

all_croppedims = all_camims[:, :, win[0]:win[1], win[2]:win[3]]

imfluxes = np.sum(all_croppedims, axis=(2,3))
eltimes = all_imtimes - all_startimes

plt.clf()
plt.plot((all_imtimes[0,:] - all_startimes)*1e3)

title = 'Flip between flat & 120 P-V stripes'
plt.clf()
plt.subplot(2,1,1)
plt.title(title)
m = [1,3,5,7]
plt.plot(eltimes[:,m]*1e3, imfluxes[:,m], '-+')
plt.xlabel('Time after call to get_n_images (ms)')
plt.xlim(0,50)
plt.subplot(2,1,2)
m = [2,4,6]
plt.plot(eltimes[:,m]*1e3, imfluxes[:,m], '-+')
plt.xlabel('Time after call to get_n_images (ms)')
plt.xlim(0,50)
plt.tight_layout()


# eltime = all_imtimes- all_imtimes[0]
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


# offset = 0
# for k in range(2): #all_camims.shape[0]):
#     plt.clf()
#     plt.imshow(all_croppedims)
#     plt.title(k+offset)
#     plt.pause(1)

