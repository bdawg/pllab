import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from astropy.io import fits
import numpy as np
from matplotlib.patches import Circle

datapath = '/Users/bnorris/Dropbox/bn/PLlab_data/snapshots/'
filename = 'buffer_11042023_160129_backref-labseeing_newalign_wiggling_copy.fit'
# filename = 'buffer_11042023_160129_backref-labseeing.fit'

# incube = fits.getdata(datapath+filename, memmap=True)

hdulist = fits.open(datapath+filename, memmap=True)
incube = hdulist[0].data

win = [252, 292, 336, 376]
croppedcube = incube[:, win[0]:win[1], win[2]:win[3]]

start = 4657
stop = 5257 #257

start = 2350
stop = 3100

step = 10
waittime = 0.1

# normcube = []
# for ind in np.arange(start, stop, step):
#     # normcube.append(croppedcube[ind, :, :] / np.sum(croppedcube[ind, :, :]))
#     plt.clf()
#     plt.imshow(croppedcube[ind, :, :])
#     plt.title(ind)
#     plt.pause(waittime)
# # normcube = np.array(normcube)
# # summedcube = np.mean(normcube, axis=0)

summedcube = np.mean(croppedcube[start:stop, :, :], axis=0)
summedcube -= (np.min(summedcube) - 1e-6)


plt.clf()
plt.imshow(summedcube)#, clim=[0,1000])
# plt.imshow(np.log10(summedcube), vmin=2)
core_rad_px = 13
mf_edge = Circle( ((win[1]-win[0])/2, (win[3]-win[2])/2), core_rad_px, fill=False, edgecolor='w',
                  linestyle='--')
plt.gca().add_artist(mf_edge)
plt.colorbar()



# plt.imshow(summedcube_both+3*summedcube_backillum)
# plt.imshow(summedcube_backillum)




"""
FWHM of central spot measured to be ~8.6 pixels
Diameter of coure ~26 pixels
"""


