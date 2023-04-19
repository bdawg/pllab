import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import *
from plcams import *
import matplotlib
matplotlib.use('TkAgg')


# lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'

darkfile1 = 'dark_psfcam_50us'

slm = plslm(lutfile=lutfile)
cam1 = credcam(camera_index=1, darkfile=darkfile1, verbose=True)
# cam1 = credcam(camera_index=2, darkfile=darkfile1, verbose=True)
cam2 = credcam(camera_index=1, darkfile=darkfile1, verbose=True)
cam2.send_command('set tint 1')


cnt1 = (257, 340)
cnt2 = (274, 357)
wsz = 96

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


nloops = 4
n_imstolog = 30#300

slm.slmwrite(all_slmimages[0, :, :], showplot=False)
time.sleep(0.5)

cam1.set_nims_tolog(n_imstolog)
all_camims1 = np.zeros((n_imstolog, nloops*len(all_slmimages), cam1.camdims[1],
                        cam1.camdims[0]), dtype=np.int16)
all_imtimes1 = np.zeros((n_imstolog, nloops*len(all_slmimages)))
all_startimes1 = np.zeros(nloops*len(all_slmimages))

cam2.set_nims_tolog(n_imstolog)
all_camims2 = np.zeros((n_imstolog, nloops*len(all_slmimages), cam2.camdims[1],
                        cam2.camdims[0]), dtype=np.int16)
all_imtimes2 = np.zeros((n_imstolog, nloops*len(all_slmimages)))
all_startimes2 = np.zeros(nloops*len(all_slmimages))


count = 0
for k in range(nloops):
    for l in range(len(all_slmimages)):
        slmim = all_slmimages[l, :, :]
        slm.slmwrite(slmim, showplot=False)

        all_startimes1[count] = time.perf_counter()
        cam1.get_n_images(blocking=False)
        all_startimes2[count] = time.perf_counter()
        cam2.get_n_images(blocking=False)

        while cam1.nims_lefttolog > 0:
            time.sleep(0.1)
        camim1 = cam1.loggedims_cube
        camim2 = cam2.loggedims_cube
        print('Got all images.')

        camim1 = camim1 - cam1.dark
        all_camims1[:, count, :, :] = camim1
        all_imtimes1[:, count] = cam1.loggedims_times_arr
        camim2 = camim2 - cam2.dark
        all_camims2[:, count, :, :] = camim2
        all_imtimes2[:, count] = cam2.loggedims_times_arr

        count += 1

print('Done')

win1 = (cnt1[0]-wsz//2, cnt1[0]+wsz//2-1, cnt1[1]-wsz//2, cnt1[1]+wsz//2-1)
all_croppedims1 = all_camims1[:, :, win1[0]:win1[1], win1[2]:win1[3]]
# win2 = (cnt2[0]-wsz//2, cnt2[0]+wsz//2-1, cnt2[1]-wsz//2, cnt2[1]+wsz//2-1)
win2 = win1
all_croppedims2 = all_camims2[:, :, win2[0]:win2[1], win2[2]:win2[3]]


# m = 1
# for k in range(100):
#     plt.clf()
#     plt.imshow(all_croppedims2[k, m, :, :])
#     plt.title(k)
#     plt.pause(0.001)

imfluxes1 = np.sum(all_croppedims1, axis=(2,3))
eltimes1 = all_imtimes1 - all_startimes1
# eltimes1 = all_imtimes1 - all_imtimes1[0,:]

imfluxes2 = np.sum(all_croppedims2, axis=(2,3))
eltimes2 = all_imtimes2 - all_startimes2
# eltimes2 = all_imtimes2 - all_imtimes2[0,:]


# plt.clf()
# plt.plot((all_imtimes2[0,:] - all_startimes2)*1e3)

eltimes = eltimes1
imfluxes = imfluxes1

# title = 'Flip between flat & 120 P-V stripes'
plt.clf()
plt.subplot(2,1,1)
# plt.title(title)
m = [1,3,5,7]
plt.plot(eltimes[:,m]*1e3, imfluxes[:,m], '-+')
# plt.plot(imfluxes[:,m], '-+')
plt.xlabel('Time after call to get_n_images (ms)')
# plt.xlim(0,50)
plt.subplot(2,1,2)
m = [2,4,6]
plt.plot(eltimes[:,m]*1e3, imfluxes[:,m], '-+')
# plt.plot(imfluxes[:,m], '-+')
plt.xlabel('Time after call to get_n_images (ms)')
# plt.xlim(0,50)
plt.tight_layout()

plt.clf()
plt.plot(eltimes1[:,0]*1e3, '-x')
plt.plot(eltimes2[:,0]*1e3, '-x')
plt.xlabel('Frame no.')
plt.ylabel('time (ms)')


