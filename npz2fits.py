import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
plt.ion()

ddir = '/Users/bnorris/DontBackup/PL/202406/'
file = 'pllabdata_20240606_randsrc2_seeing-rep10_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand-flatn10_10K_01_file00.npz'

# ddir = 'C:/Data/'
# file = 'pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_file00.npz'

arrnames = ['imcube_cam0', 'imcube_cam1']
arrinds = [0, 1]

first_n = None
first_n = 1000

outdir = ddir

npf = np.load(ddir+file, allow_pickle=True)
for k in range(len(arrnames)):
    imcube = npf[arrnames[k]]
    dk = dk = npf['darkframes'][k]
    imcube = imcube - dk
    imcube[:,0,:] = 0

    if first_n is not None:
        imcube = imcube[:first_n,:,:]

    outfilename = file[:-4] + '_cam%d' % k + '.fits'
    fits.writeto(outdir+outfilename, imcube)

# data.shape
# plt.figure(1)
# plt.clf()
# plt.imshow(data[1000,:,:])

# fits.writeto(outdir+'out.fits', data)