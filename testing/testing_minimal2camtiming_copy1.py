
import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')
import FliSdk_V2


cam_index0 = 1
cam_context0 = FliSdk_V2.Init()
listOfGrabbers = FliSdk_V2.DetectGrabbers(cam_context0)
listOfCameras = FliSdk_V2.DetectCameras(cam_context0)
num_cameras = len(listOfCameras)
print('Using camera ' + listOfCameras[cam_index0])
errorval = FliSdk_V2.SetCamera(cam_context0, listOfCameras[cam_index0])
FliSdk_V2.SetMode(cam_context0, FliSdk_V2.Mode.Full) # Enables grab and config
errorval = FliSdk_V2.Update(cam_context0)
FliSdk_V2.Start(cam_context0)

# cam_index1 = 1
# cam_context1 = FliSdk_V2.Init()
# listOfGrabbers = FliSdk_V2.DetectGrabbers(cam_context1)
# listOfCameras = FliSdk_V2.DetectCameras(cam_context1)
# print('Using camera ' + listOfCameras[cam_index1])
# num_cameras = len(listOfCameras)
# errorval = FliSdk_V2.SetCamera(cam_context1, listOfCameras[cam_index1])
# FliSdk_V2.SetMode(cam_context1, FliSdk_V2.Mode.Full) # Enables grab and config
# errorval = FliSdk_V2.Update(cam_context1)
# FliSdk_V2.Start(cam_context1)

def goodtimer(time_ms):
    tm = time_ms / 1000
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < tm:
        pass


n_ims = 1000
waittime_ms = 10

# loggedims_cube0 = np.zeros((n_ims, 56, 64))
loggedims_cube0 = np.zeros((n_ims, 64, 64))
loggedims_cube1 = np.zeros((n_ims, 64, 64))
loggedims_times_arr = np.zeros(n_ims)

for k in range(n_ims):
    if k % 100 == 0:
        print(k)
    goodtimer(waittime_ms)

    new_im0 = FliSdk_V2.GetRawImageAsNumpyArray(cam_context0, -1)
    # goodtimer(waittime_ms)
    # new_im1 = FliSdk_V2.GetRawImageAsNumpyArray(cam_context1, -1)

    loggedims_cube0[k, :, :] = new_im0
    # loggedims_cube1[k, :, :] = new_im1

    loggedims_times_arr[k] = time.perf_counter()

# loggedims_cube0[:,0,:] = 0
# loggedims_cube1[:,0,:] = 0

flux0 = np.mean(loggedims_cube0, (1, 2))
flux1 = np.mean(loggedims_cube1, (1, 2))
dflux0 = np.diff(flux0)
dflux1 = np.diff(flux1)

plt.clf()
plt.subplot(211)
plt.plot(dflux0, '-+')
plt.subplot(212)
plt.plot(dflux1, '-+')

nzeros = np.sum(dflux0 == 0)# + np.sum(dflux1 == 0)
print('Number of dflux=0s: %d' % nzeros)

# Check with actual pixel differences
diffims = np.diff(loggedims_cube0, axis=0)
fluxdifims = np.sum(diffims, (1, 2))
ndiffimzeros = np.sum(fluxdifims == 0)
print('Number of diffim_sums=0s: %d' % ndiffimzeros)



# plt.clf()
# plt.plot(np.diff(loggedims_times_arr))

# for k in range(n_ims):
#     plt.clf()
#     plt.subplot(211)
#     plt.imshow(loggedims_cube0[k,:,:])
#     plt.subplot(212)
#     plt.imshow(loggedims_cube1[k,:,:])
#     plt.pause(0.5)

