import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')

def goodtimer(time_ms):
    tm = time_ms / 1000
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < tm:
        pass



cam = credcam(camera_index=1, verbose=True)

n_imstolog = 1000

# all_camims = np.zeros((n_imstolog, cam.camdims[1], cam.camdims[0]), dtype=np.int16)
# all_imtimes = np.zeros(n_imstolog)

# camims = cam.get_n_images(return_ims=True, coadd=False, subtract_dark=False)

cam.set_nims_tolog(n_imstolog)
cam.get_n_images(blocking=False)
while cam.nims_lefttolog > 0:
    time.sleep(0.1)
camims = cam.loggedims_cube

# cam.send_command('set swsynchro off')
# cam.send_command('set swsynchro source swtrig')
# cam.send_command('set nbframesperswtrig 1')
# waittime_ms = 10
# cam.set_nims_tolog(n_imstolog)
# cam.get_n_images(blocking=False)
# for k in range(n_imstolog):
#     cam.send_command('swtrig')
#     goodtimer(waittime_ms)
# camims = cam.loggedims_cube



plt.clf()
plt.plot(np.diff(cam.loggedims_times_arr), '-+')

fluxes = np.mean(camims, (1, 2))
dflux = np.diff(fluxes)
# plt.clf()
# plt.plot(dflux, '-+')
nzeros = np.sum(dflux == 0)# + np.sum(dflux1 == 0)
print('Number of dflux=0s: %d' % nzeros)

# Check with actual pixel differences
diffims = np.diff(camims, axis=0)
fluxsqdifims = np.sum(diffims**2, (1, 2))
ndiffimzeros = np.sum(fluxsqdifims == 0)
print('Number of diffim_sums=0s: %d' % ndiffimzeros)


# camims[:,0,:] = 0
# diffims = np.diff(camims, axis=0)
# # diffims = camims - camims[0,:,:]
# diffims = camims
#
# for k in range(190, 194):
#     plt.clf()
#     plt.imshow(diffims[k,:,:])
#     plt.title('%d'%k + ' sum: %.2f' % (np.sum(diffims[k,:,:])))
#     plt.pause(1)



