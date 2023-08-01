import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.animation import FFMpegWriter
from astropy.io import fits
from PIL import Image

# datapath = '/Users/bnorris/Dropbox/bn/PLlab_data/snapshots/'
datapath = '/Users/bnorris/DontBackup/PL/202306/'
darkfilename = '15062023_170128_backref_dark.fit'

datafilename = '15062023_170128_backref_backillum_superK.fit'
# datafilename = '15062023_170128_backref_backillumsuperk_PLUSlaserPSF.fit'
datafilename = 'plcube_pllabdata_20230605_superK_slmcube_20230505_seeing_0.4-10_10K_01_file00.fits'

outfilename = '20230605_superK_slmcube_20230505_seeing_0.4-10_10K_01_spd2.mp4'
save_movie = True

nframes = None
# nframes = 100

showdiffs = False

speedup = 25
# speedup = 2000 / 25
# speedup = 1000 / 25

# frm_av = 20
frm_av=1

cropsize = 64
cnt = [288,389]
# cropsize = 80 #200
# cnt = [273,357]


# clim=[-3,0]
# clim=[-0.15,0.8]
clim=[0,1]
clim = None
cmap = 'viridis' #'inferno'
# cmap = 'inferno'
power = 1
# power = 3
bias = 0


if not 'cube' in locals():
    # hdulist = fits.open(datapath + darkfilename)
    # cube = hdulist[0].data
    # header = hdulist[0].header
    # darkframe = np.mean(cube, 0)
    # # darkframe = 0

    hdulist = fits.open(datapath+datafilename)
    cube = hdulist[0].data
    header = hdulist[0].header
    cleancube = cube# - darkframe
    # cleancube[:,0,:] = 0 # Hack - zero the first row of each frame to get rid of clock
if nframes is None:
    nframes = cube.shape[0]


writer = FFMpegWriter(fps=25)
frames = np.arange(0, nframes, speedup)
# frames = np.arange(nframes, 0, -speedup)
# frames = frames[1:]
sz = cropsize // 2
fig = plt.figure(3, figsize=(6, 5))

# frames = np.random.randint(0, nframes, size=nframes//10)

# frames = np.hstack((frames, np.flip(frames)))
# for k in range(3):
#     frames = np.hstack((frames,frames))

if showdiffs:
    meanim = np.mean(cleancube,axis=0)
    meanim = meanim / np.max(meanim)
    all_meanims = []

# # Test plot:
# frm = 0
# im = cleancube[frm, :, :]
# im = np.mean(cleancube[frm:frm+int(frm_av),:,:],0)
# im = im + bias
# im = im/np.max(im)
# # im = im + 1e-2
# # im[im<1e-15] = 1e-15
# # im = np.log10(im)
# plt.clf()
# plt.imshow(im[cnt[0] - sz:cnt[0] + sz - 1, cnt[1] - sz:cnt[1] + sz - 1]**power, clim=clim, cmap=cmap)
# plt.colorbar()
# plt.tight_layout()


if save_movie is False:
    outfilename = 'temp.mp4'
count = 0
all_croppedims = []
with writer.saving(fig, outfilename, 100):
    for frm in frames:
        frm = int(frm)
        im = np.mean(cleancube[frm:frm + frm_av, :, :], 0)
        im = im + bias
        im = im / np.max(im)
        # im = np.log10(im)
        if showdiffs:
            im = im - meanim
            all_meanims.append(im[cnt[0] - sz:cnt[0] + sz - 1, cnt[1] - sz:cnt[1] + sz - 1])
        plt.clf()
        # imsh = im[cnt[0] - sz:cnt[0] + sz - 1, cnt[1] - sz:cnt[1] + sz - 1]**power
        imsh = im ** power
        all_croppedims.append(imsh)
        plt.imshow(imsh, clim=clim, cmap=cmap)
        plt.title(count)
        # plt.colorbar()
        plt.tight_layout()
        if save_movie:
            writer.grab_frame()
        else:
            plt.pause(0.01)

        if count % 100 == 0:
            print(count)
        count += 1

print('Done.')


# from matplotlib import cm
# # Image.fromarray(all_croppedims[3]).save('imsave.tif')
# # im_psf = all_croppedims[3]
# im_psf_col = cm.viridis(im_psf)
#
# im_fib = all_croppedims[30]
# im_fib_col = cm.gray(im_fib)
# a = 0.5
# b = 0.5
# plt.imshow(a*im_fib_col + b*im_psf_col)
