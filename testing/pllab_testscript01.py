import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from plcams import *

cam = credcam(camera_index=1, verbose=True)

# Get n images
n_ims = 100
cam.set_nims_tolog(n_ims)

starttime = time.perf_counter()
imcube = cam.get_n_images(return_ims=True).astype('float32')
print('Time to start: %f' % (cam.loggedims_times_arr[0] - starttime))
print('Total time: %f' % (cam.loggedims_times_arr[-1] - starttime))


### Test non-blocking
n_ims = 1800
cam.set_nims_tolog(n_ims)
starttime = time.perf_counter()
cam.get_n_images(blocking=False)
while cam.nims_lefttolog > 0:
    time.sleep(0.1)
print('Got all images.')
imcube = cam.loggedims_cube
print('Time to start: %f' % (cam.loggedims_times_arr[0] - starttime))
print('Total time: %f' % (cam.loggedims_times_arr[-1] - starttime))



# cnt = (257, 340)
# wsz = 96
# win = (cnt[0]-wsz//2, cnt[0]+wsz//2-1, cnt[1]-wsz//2, cnt[1]+wsz//2-1)
#
#
# plt.figure(1)
# # immns = np.mean(imcube, axis=(1,2))
# # immns = np.std(imcube-imcube[0,:,:], axis=(1,2))
# immns = np.std(imcube[:,win[0]:win[1], win[2]:win[3]]-imcube[0,win[0]:win[1],
#                                                       win[2]:win[3]], axis=(1,2))
# plt.clf()
# plt.plot(immns)
#
# d = np.diff(immns)
# print(np.where(d==0))

tdiff = np.diff(cam.loggedims_times_arr)
plt.figure(2)
plt.clf()
plt.plot(tdiff)
plt.pause(0.001)


# plt.imshow(imcube[0, :, :])

# offset = 0
# dk = imcube[0,:,:].astype('float32')
# plt.figure(3)
# for k in range(10):#imcube.shape[0]):
#     plt.clf()
#     plt.imshow(imcube[k+offset,win[0]:win[1], win[2]:win[3]]-dk[win[0]:win[1],
#                                                              win[2]:win[3]], clim=[-80,80])
#     plt.title(k+offset)
#     # plt.imshow(imcube[k+0, :, :] - dk[:, :])
#     plt.pause(0.001)


# t = time.perf_counter()
# for k in range(1000000):
#     pass
# print((time.perf_counter()-t)/1000000)

