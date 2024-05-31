import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
plt.ion()

ddir = '/Users/bnorris/DontBackup/PL/202306/'
file = 'pllabdata_20230617e_superK_slmcube_20230505_seeing_0.4-10_10K_01_file00.npz'
arrname = 'imcube_cam1'

# file ='slmcube_20230505_seeing_0.4-10_10K_01_file00.npz'
# arrname = 'all_slmims'

ddir = '/Volumes/bnorris/Data/PL/202306-seeingdata/'
file = 'pllabdata_20230617a_laser_slmcube_20230505_seeing_0.4-10_10K_01_file09.npz'

outdir = ddir
outdir = './'

npf = np.load(ddir+file)
data = npf[arrname]

# data.shape
# plt.figure(1)
# plt.clf()
# plt.imshow(data[1000,:,:])

fits.writeto(outdir+'out.fits', data)