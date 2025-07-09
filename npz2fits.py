import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
plt.ion()

outfile_suff = ''

ddir = '/Users/bnorris/DontBackup/PL/202406/'
file = 'pllabdata_20240606_randsrc2_seeing-rep10_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand-flatn10_10K_01_file00.npz'

# ddir = 'C:/Data/'
# file = 'pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_file00.npz'

ddir = '/Users/bnorris/DontBackup/PL/202209_scexaoPL/analysis202410_combineddataset/'
file = 'preprocdata_20220928_VegaSet01_01_ircam0_stride4.npz'

arrnames = ['imcube_cam0', 'imcube_cam1']
remove_clock = True

arrnames = ['cube']
remove_clock = False
outfile_suff = '_extraStride8'

first_n = None
# first_n = 100

# start_ind = 0
# end_ind = None

stride = None # None to keep every frame. Applied after first_n.
stride = 8

outdir = ddir

npf = np.load(ddir+file, allow_pickle=True)
for k in range(len(arrnames)):
    imcube = npf[arrnames[k]]
    print('Input file contains %d frames' % imcube.shape[0])
    try:
        dk = dk = npf['darkframes'][k]
    except:
        print('WARNING: No darkframes found')
        dk = 0
    imcube = imcube - dk
    if remove_clock:
        imcube[:,0,:] = 0

    if first_n is not None:
        imcube = imcube[:first_n,:,:]

    if stride is not None:
        print('Using stride of %d' % stride)
        newinds = np.arange(0, imcube.shape[0], stride)
        imcube = imcube[newinds, :, :]

    outfilename = file[:-4] + '_cam%d' % k + outfile_suff + '.fits'
    fits.writeto(outdir+outfilename, imcube)

# data.shape
# plt.figure(1)
# plt.clf()
# plt.imshow(data[1000,:,:])

# fits.writeto(outdir+'out.fits', data)