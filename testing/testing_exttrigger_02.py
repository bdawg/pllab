import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')
plt.ion()


datadir = '../pllab_data/'
# slmims_filename = 'slmcube_varyingstripes_0-60_10frm_01.npz'
slmims_filename = 'slmcube_alternatingstripes_0-60_10frm_01.npz'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'
darkfiledir = '../pllab_data/darks/'
darkfile1 = None #'dark_psfcam_50us'

verbose = True
nloops = 10
# For PSFcam:
cnt1 = (257, 340)
wsz1 = 48
# For PLcam:
cnt2 = (160, 282)
wsz2 = 120

cam_syncdelay_ms = 12
extra_delay_ms  = 3
wait_time_ms = cam_syncdelay_ms + extra_delay_ms

if not 'slm' in locals():
    slm = plslm(lutfile=lutfile)
    cam1 = credcam(camera_index=1, darkfile=darkfile1, darkpath=darkfiledir, verbose=verbose)
    cam1.set_tint(0.00005, verbose=verbose)
    cam1.external_trigger(enabled=True, syncdelay=cam_syncdelay_ms, verbose=verbose)
    cam2 = credcam(camera_index=0, darkfile=darkfile1, darkpath=darkfiledir, verbose=verbose)
    cam2.set_tint(0.0005, verbose=verbose)
    cam2.external_trigger(enabled=True, syncdelay=cam_syncdelay_ms, verbose=verbose)

slmimdataf = np.load(datadir + slmims_filename)
slmims = slmimdataf['all_slmims']
if type(slmims[0, 0, 0]) is not np.int8:
    print('Error: input SLM cube not int8')
all_slmim_params = slmimdataf['all_slmim_params']


n_slmims = slmims.shape[0]
win1 = (cnt1[0]-wsz1//2, cnt1[0]+wsz1//2-1, cnt1[1]-wsz1//2, cnt1[1]+wsz1//2-1)
win2 = (cnt2[0]-wsz2//2, cnt2[0]+wsz2//2-1, cnt2[1]-wsz2//2, cnt2[1]+wsz2//2-1)

# croppeddark = cam.dark[win[0]:win[1], win[2]:win[3]]
total_nims = nloops * n_slmims

slm.slmwrite(slmims[0, :, :], showplot=False)
time.sleep(0.1)

# cam1.reset_buffer()
# cam2.reset_buffer()
# time.sleep(1)

cam1.set_nims_tolog(total_nims)
cam2.set_nims_tolog(total_nims)
# time.sleep(1)


print('Starting data acquisition')
startime = time.time()
cam1.get_n_images(blocking=False, return_ims=False, coadd=False, subtract_dark=False)
cam2.get_n_images(blocking=False, return_ims=False, coadd=False, subtract_dark=False)

for k in range(nloops):
    for l in range(n_slmims):
        slmim = slmims[l, :, :]
        slm.slmwrite(slmim, showplot=False, skip_readycheck=True)

        cam1.goodtimer(wait_time_ms)

print('Done - elapsed time %.2f seconds' % (time.time() - startime))

all_camims1 = cam1.loggedims_cube
all_imtimes1 = cam1.loggedims_times_arr
# all_camims1 = all_camims1 - cam1.dark
all_camims1 = all_camims1[:, win1[0]:win1[1], win1[2]:win1[3]]
imfluxes1 = np.sum(all_camims1, axis=(1, 2))
eltime1 = all_imtimes1 - all_imtimes1[0]

all_camims2 = cam2.loggedims_cube
all_imtimes2 = cam2.loggedims_times_arr
# all_camims2 = all_camims2 - cam2.dark
all_camims2 = all_camims2[:, win2[0]:win2[1], win2[2]:win2[3]]
imfluxes2 = np.sum(all_camims2, axis=(1, 2))
eltime2 = all_imtimes2 - all_imtimes2[0]


eltime = eltime1
imfluxes = imfluxes1
all_imtimes = all_imtimes1
# eltime = eltime2
# imfluxes = imfluxes2
# all_imtimes = all_imtimes2

# plt.clf()
# plt.plot(all_imtimes1)

xlim = [0,0.5]
xlim = None
plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.title('cam_syncdelay_ms: %.1f, extra_delay_ms: %.1f' % (cam_syncdelay_ms, extra_delay_ms))
# plt.plot(eltime, imfluxes, '-x')
plt.plot(imfluxes, '-x')
plt.xlim(xlim)
plt.ylabel('Window flux')
plt.subplot(2,1,2)
# plt.plot(eltime,'-x')
# plt.ylabel('Frame time from start (s)')
plt.plot(np.diff(all_imtimes*1000),'-x')
plt.ylabel('Time between frames (ms)')
plt.tight_layout()


# all_eltimes = []
# all_imfluxes = []
# all_delays = []

# all_eltimes.append(eltime)
# all_imfluxes.append(imfluxes)
# all_delays.append([cam_syncdelay_ms, extra_delay_ms])

# all_eltimes_a = np.array(all_eltimes)
# all_imfluxes_a = np.array(all_imfluxes)
# all_delays_a = np.array(all_delays)
# np.savez('timingtests_20230503_10stepstripes_varyextracamsyncdelay.npz', all_eltimes_a=all_eltimes_a,
#          all_imfluxes_a=all_imfluxes_a, all_delays_a=all_delays_a)

# plt.figure(3)
# plt.clf()
# plt.plot(all_imfluxes_a.T[:,5:15], '-+', alpha=0.5)
# plt.xlim(0,22)
# plt.legend(all_delays_a[5:15,0], loc='upper right')
#
# plt.clf()
# plt.plot(np.diff(all_eltimes_a).T, '-+', alpha=0.5)
# plt.ylim(0,0.03)
# plt.ylim(0.013,0.020)
# # plt.xlim(0,10)
# plt.legend(all_delays_a[:,1], loc='upper center')#loc='upper right')





# plt.figure(2)
# plt.clf()
# plt.subplot(121)
# plt.imshow(all_camims[-2,:,:])
# plt.colorbar()
# plt.subplot(122)
# plt.imshow(all_camims[-1,:,:])
# plt.colorbar()

# plt.figure(2)
# for ind in range(10):
#     plt.clf()
#     plt.imshow(all_camims[ind,:,:], vmax=10000)
#     plt.colorbar()
#     plt.pause(0.01)

# plt.figure(2)
# for ind in range(10):
#     plt.clf()
#     plt.imshow(slmims[ind,:,:], clim=[-130,130])
#     plt.colorbar()
#     plt.title(ind)
#     plt.pause(1)


