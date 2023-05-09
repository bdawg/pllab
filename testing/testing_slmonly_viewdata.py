import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
from astropy.io import fits

datadir = '../pllab_data/manual_saves/'
datadir = 'C:/Users/sail/Desktop/data_tempsave/'
psfcam_filename = 'buffer_03052023_181306_psf.fit'
plcam_filename = 'buffer_03052023_181322_pl.fit'


# For PSFcam:
cnt1 = (257, 340)
wsz1 = 48
# For PLcam:
cnt2 = (160, 282)
wsz2 = 120


all_camims1 = fits.getdata(datadir+psfcam_filename)
all_camims2 = fits.getdata(datadir+plcam_filename)

win1 = (cnt1[0]-wsz1//2, cnt1[0]+wsz1//2-1, cnt1[1]-wsz1//2, cnt1[1]+wsz1//2-1)
win2 = (cnt2[0]-wsz2//2, cnt2[0]+wsz2//2-1, cnt2[1]-wsz2//2, cnt2[1]+wsz2//2-1)

all_camims1 = all_camims1[:, win1[0]:win1[1], win1[2]:win1[3]]
imfluxes1 = np.sum(all_camims1, axis=(1, 2))
all_camims2 = all_camims2[:, win2[0]:win2[1], win2[2]:win2[3]]
imfluxes2 = np.sum(all_camims2, axis=(1, 2))

# imfluxes = imfluxes1
# imfluxes = imfluxes2

plt.figure(1)
plt.clf()
xlim = [0,20]
xlim = None
# plt.title('cam_syncdelay_ms: %.1f, extra_delay_ms: %.1f' % (cam_syncdelay_ms, extra_delay_ms))
plt.subplot(211)
plt.title('psfcam')
plt.plot(imfluxes1, '-x')
plt.xlim(xlim)
plt.ylabel('Window flux')
plt.subplot(212)
plt.title('plcam')
plt.plot(imfluxes2, '-x')
plt.xlim(xlim)
plt.ylabel('Window flux')


