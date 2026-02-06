import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.io import fits

datadir = '/Users/bnorris/DontBackup/PL_data/'
infile = 'psfcube_pllabdata_20230616_laser_01_slmcube_varyingstripes_0-127_01.fits'
cnt = (138, 138)

# infile = 'psfcube_pllabdata_20230615_slmcube_varyingstripes_0-127_01.fits'
# cnt = (34, 34)
# cnt = (200, 200) # dark
sz = 10

# infile=('plcube_pllabdata_20230616_laser_01_slmcube_varyingstripes_0-127_01.fits')
# cnt = (79, 109)
# sz = 50

cube = fits.getdata(datadir + infile)

# k=0
# im = cube[k, :, :]
# subim = im[cnt[0]-sz:cnt[0]+sz, cnt[1]-sz:cnt[1]+sz]
# plt.imshow(subim)

nims = cube.shape[0]
fluxes = np.zeros(nims)
for k in range(nims):
    im = cube[k, :, :]
    subim = im[cnt[0]-sz:cnt[0]+sz, cnt[1]-sz:cnt[1]+sz]
    fluxes[k] = np.sum(subim)

    # plt.clf()
    # plt.imshow(subim)
    # plt.pause(0.001)


# Show single period

# fluxes = fluxes[130:]
fluxes = fluxes[3:129]
fluxes = np.flip(fluxes)
rads = np.linspace(0, 1, fluxes.shape[0]) * 2 * np.pi
fluxes = fluxes / np.max(fluxes)

plt.plot(rads, fluxes)
plt.ylim(0, None)
plt.xlabel('Grating phase P-V (rad)')
plt.ylabel('Total flux')

print(np.mean(fluxes))
# normfact = np.max(fluxes)
max_flux = np.max(fluxes) #/ normfact
min_flux = np.min(fluxes) #/ normfact

print('Min flux fraction: %.3f' % min_flux)

plt.title('Min flux = %.3f' % min_flux)
#
# slmcube_file = 'slmcube_varyingstripes_0-127_01.npz'
# npf_slm = np.load(datadir+slmcube_file)
# all_slmims = npf_slm['all_slmims']
# plt.figure(2)
# # plt.plot(all_slmims[64, 512, :])
# slm_pv = np.max(all_slmims, axis=(1,2)) - np.min(all_slmims, axis=(1,2))
# plt.plot(slm_pv)
#
# slmims2 = np.copy(all_slmims)
# for k in range(127):
#     if slmims2[k] > 127:
#         slmims2[k] = slmims2(k) - 255
# slm2_pv = np.max(slmims2, axis=(1,2)) - np.min(slmims2, axis=(1,2))
# plt.plot(slm2_pv)
