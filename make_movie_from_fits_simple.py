import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.animation import FFMpegWriter
from astropy.io import fits
from PIL import Image

datapath = '/Users/bnorris/Dropbox/bn/PLlab_data/snapshots/'
darkfilename = '15062023_170128_backref_dark.fit'

datafilename = '15062023_170128_backref_backillum_superK.fit'


outfilename = '20230605_superK_slmcube_20230505_seeing_0.4-10_10K_01_spd2.mp4'
save_movie = True

nframes = None
# nframes = 100

showdiffs = False

speedup = 25

# frm_av = 20
frm_av=1

cropsize = 64
cnt = [288,389]


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
    hdulist = fits.open(datapath+datafilename)
    cube = hdulist[0].data
    header = hdulist[0].header
    cleancube = cube# - darkframe
    # cleancube[:,0,:] = 0 # Hack - zero the first row of each frame to get rid of clock
if nframes is None:
    nframes = cube.shape[0]


writer = FFMpegWriter(fps=25)
frames = np.arange(0, nframes, speedup)
sz = cropsize // 2
fig = plt.figure(3, figsize=(6, 5))


if showdiffs:
    meanim = np.mean(cleancube,axis=0)
    meanim = meanim / np.max(meanim)
    all_meanims = []


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

