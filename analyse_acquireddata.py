import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from astropy.io import fits
import numpy as np
import os

# datadir = '../pllab_data/'
datadir = 'C:/Data/'
datadir = '/Users/bnorris/DontBackup/PL/202306/'
datafilename = 'pllabdata_20230616_laser_slmcube_20230505_zerns_25modes_0.4_10K_01_file00.npz'

slmims_filename = None
# slmims_filename = datadir+'slmcube_20230505_zerns_25modes_0.5-0.1_10K_01_file00.npz'

window_ims = True
# winparams = [[257, 340, 512],
#              [160, 282, 120]]
winparams = [[140, 145, 64],
             [80, 90, 120]]
winparams = [[140, 145, 64],
             [75, 115, 120]]

winparams_fluxsum = [[138, 138, 42],  # Crop mode
                     [88, 77, 120]]
window_slmims = False
slmwinparams = [553, 378, 400]

# logpsf = True
# logpl = False

fits_suffix = ''
save_psf_fits = True
save_pl_fits = True
# save_psf_fits = False
# save_pl_fits = False

measure_fluxsum = True

npf = np.load(datadir+datafilename, allow_pickle=True)
imcube_psf = npf['imcube_cam0']
imcube_pl = npf['imcube_cam1']
try:
    darkframes = npf['darkframes']
    imcube_psf = imcube_psf - darkframes[0]
    imcube_pl = imcube_pl - darkframes[1]
except:
    print('Warning - no dark frames found')

# dknpf = np.load('../pllab_data/darks_20230603b_laser.npz', allow_pickle=True)
# darkframes = dknpf['darkframes']
if slmims_filename is None:
    slmims_filename = npf['slmims_filename'].item()

slmimdataf = np.load(datadir + os.path.split(slmims_filename)[-1], allow_pickle=True)
# slmimdataf = np.load(slmims_filename, allow_pickle=True)

slmims = slmimdataf['all_slmims']
# slmims = slmimdataf['array']
slmim_params = slmimdataf['all_slmim_params']
# print(slmim_params)
nslm = slmims.shape[0]


nshow = 10

if window_ims:
    wins = []
    for winparam in winparams:
        wsz = winparam[2]
        win = (winparam[0] - wsz // 2, winparam[0] + wsz // 2, winparam[1] - wsz // 2,
               winparam[1] + wsz // 2)
        wins.append(win)
    k = 0
    imcube_psf = imcube_psf[:, wins[k][0]:wins[k][1], wins[k][2]:wins[k][3]]
    k = 1
    imcube_pl = imcube_pl[:, wins[k][0]:wins[k][1], wins[k][2]:wins[k][3]]

if window_slmims:
    slmwsz = slmwinparams[2]
    slmwin = (slmwinparams[0] - slmwsz // 2, slmwinparams[0] + slmwsz // 2, slmwinparams[1] - slmwsz // 2,
           slmwinparams[1] + slmwsz // 2)
    slmims = slmims[:, slmwin[0]:slmwin[1], slmwin[2]:slmwin[3]]

def logim(im, loclip=-2.5):
    im -= np.min(im)
    im /= np.max(im)
    im = np.log10(im)
    im = np.clip(im, a_min=loclip, a_max=None)
    return im

plt.figure(1)
for show_ind in np.arange(0,nshow,10):
    # show_ind = 0
    plt.clf()
    plt.subplot(131)
    plt.imshow(imcube_psf[show_ind,:,:])
    # plt.imshow(logim(imcube_psf[show_ind, :, :]))
    plt.title('PSF cam')
    plt.subplot(132)
    plt.imshow(imcube_pl[show_ind,:,:])
    # plt.imshow(logim(imcube_pl[show_ind, :, :]))
    plt.title('PL cam')
    plt.subplot(133)
    # plt.imshow(slmims[(show_ind % nslm),:,:])
    plt.title('SLM image')
    plt.colorbar()
    plt.pause(0.001)


if save_psf_fits:
    fitsfilename = 'psfcube_' + datafilename[:-4] + fits_suffix + '.fits'
    print('Saving PSF cube to ' + fitsfilename)
    fits.writeto(datadir+fitsfilename, imcube_psf)

if save_pl_fits:
    fitsfilename = 'plcube_' + datafilename[:-4] + fits_suffix + '.fits'
    print('Saving PL cube to ' + fitsfilename)
    fits.writeto(datadir+fitsfilename, imcube_pl)

if measure_fluxsum:
    wins = []
    ncams = 2
    for winparam in winparams_fluxsum:
        wsz = winparam[2]
        win = (winparam[0] - wsz // 2, winparam[0] + wsz // 2, winparam[1] - wsz // 2,
               winparam[1] + wsz // 2)
        wins.append(win)
    all_fluxes = []
    for k in range(ncams):
        if k == 0:
            cube = imcube_psf
        if k == 1:
            cube = imcube_pl
        # cube = cube[:, wins[k][0]:wins[k][1], wins[k][2]:wins[k][3]]
        cubefluxes = np.sum(cube, axis=(1, 2))
        all_fluxes.append(cubefluxes)

    plt.figure(3)
    plt.clf()
    plt.subplot(211)
    plt.plot(all_fluxes[0],'.', alpha=0.5)
    plt.subplot(212)
    plt.plot(all_fluxes[1],'.', alpha=0.5)


plt.figure(1)
plt.clf()
plt.imshow(imcube_psf[3,:,:])
# plt.imshow(slmims[2,:,:])
plt.colorbar()