import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
plt.ion()


infile = 'timingdata_13042023.npz'
infile = 'timingdata_17042923.npz'
infile = 'timingdata_18042023b.npz'

iminfile = 'timingdata_images_18042023b.npz'
fi = np.load(iminfile)
all_ims = fi['all_ims']

f = np.load(infile)
all_imfluxes = f['all_imfluxes']
all_eltimes = f['all_eltimes']
# wait_times = np.arange(0, 10.1, 0.1)
wait_times = f['wait_times']


# xr=[0, 20]
# allwaittimes_ms = []
# waitinds_toplot = np.arange(0, 101, 20)
# waitinds_toplot = np.arange(0, 10, 1)
# plt.figure(1)
# plt.clf()
# for ti in waitinds_toplot:
#     allwaittimes_ms.append(str(wait_times[ti]))
#     plt.plot(all_imfluxes[ti, :], '-+')
#     plt.xlim(xr[0], xr[1])
# plt.legend(allwaittimes_ms)


xr=[0, 200]
cmap = 'viridis'
aspect = 'auto'
# aspect = 'equal'
plt.figure(2)
plt.clf()
# plt.subplot(2,1,1)
plt.imshow(all_imfluxes[:, xr[0]:xr[1]], aspect=aspect, cmap=cmap, interpolation='None',
           extent=(xr[0], xr[1], np.max(wait_times), np.min(wait_times)))
# plt.imshow(all_imfluxes, aspect=aspect, cmap=cmap, interpolation='None')
plt.colorbar()
plt.xlabel('Frame number')
plt.ylabel('Wait time (ms)')
plt.title('Flux')
# plt.subplot(2,1,2)
# plt.imshow(all_eltimes*1e3, cmap=cmap)
# plt.xlabel('Frame number')
# plt.ylabel('Wait time (index)')
# plt.title('Elapsed times')
# plt.colorbar()
plt.tight_layout()

wait_ind = 35
plt.figure(5)
plt.clf()
plt.plot(all_imfluxes[wait_ind,:], 'o')
frameinds = np.arange(42,53)
plt.clf()
for k in range(len(frameinds)):
    frame = frameinds[k]
    plt.subplot(3,4,k+1)
    plt.imshow(all_ims[wait_ind, frame, :, :], clim=[0,2500])
    plt.title(frame)
    plt.colorbar()
plt.tight_layout()



### Plot combined data
# Skip first one, as left-over flux from previous cycle at short wait time
inds_switchtobright = np.arange(2, all_imfluxes.shape[1], 2)
inds_switchtodark = np.arange(1, all_imfluxes.shape[1], 2)
imfluxes_swbr = all_imfluxes[:, inds_switchtobright]
imfluxes_swdk = all_imfluxes[:, inds_switchtodark]

xr=[0, 200]
cmap = 'viridis'
plt.figure(3)
plt.clf()
plt.subplot(2,1,1)
plt.imshow(imfluxes_swbr[:, xr[0]:xr[1]], aspect='auto', cmap=cmap,
           extent=(xr[0], xr[1], np.max(wait_times), np.min(wait_times)))
plt.colorbar()
plt.xlabel('Frame number')
plt.ylabel('Wait time (ms)')
plt.title('Flux, switching to bright')
plt.subplot(2,1,2)
plt.imshow(imfluxes_swdk[:, xr[0]:xr[1]], aspect='auto', cmap=cmap,
           extent=(xr[0], xr[1], np.max(wait_times), np.min(wait_times)))
plt.colorbar()
plt.xlabel('Frame number')
plt.ylabel('Wait time (ms)')
plt.title('Flux, switching to dark')
plt.tight_layout()



mnfluxes_swbr = np.mean(imfluxes_swbr, axis=1)
mnfluxes_swdk = np.mean(imfluxes_swdk, axis=1)
plt.figure(4)
plt.clf()
# plt.subplot(2,1,1)
plt.plot(wait_times, mnfluxes_swbr, '-')
plt.plot(wait_times, mnfluxes_swdk, '-')
plt.title('Switching to bright / dark')
plt.xlabel('Wait time (ms)')
plt.ylabel('Flux')
plt.legend(['Switching to bright', 'Switching to dark'])
plt.tight_layout()
# plt.subplot(2,1,2)
# plt.plot(wait_times, mnfluxes_swdk, '-*')
# plt.title('Switching to dark')
# plt.xlabel('Wait time (ms)')
# plt.ylabel('Flux')


