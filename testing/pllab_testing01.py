import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from pllab import pllab


datadir = '../pllab_data/'
slmims_filename = 'slmcube_varyingstripes_0-60_10frm_01.npz'
nloops = 10

camstosave = ['psf', 'refl']
winparams = [ [257, 340, 42],
              [272, 358, 42]]


camsettings_psfcam =  {'sensitivity': 'low',
                  'bias mode': 'on',
                  'flat mode': 'off',
                  'badpixel mode': 'on',
                  'fps': 600,#1e6, # 1e6 Sets to maximum
                  'tint': 0.0003 # 1 Sets to maximum # 0.0002
                  }

camsettings_reflcam =  {'sensitivity': 'low',
                  'bias mode': 'on',
                  'flat mode': 'off',
                  'badpixel mode': 'on',
                  'fps': 600,#1e6, # 1e6 Sets to maximum
                  'tint': 1 # 1 Sets to maximum # 0.0002
                  }

cam_settings = [camsettings_psfcam, camsettings_reflcam]


# camstosave = ['refl']
# winparams = [[272, 358, 42]]
# cam_settings = [camsettings_reflcam]
winparams=None



pllab = pllab(datadir=datadir, camstosave=camstosave, winparams=winparams,
              cam_settings=cam_settings, verbose=True, latency=10)
pllab.load_slmims(datadir+slmims_filename)
all_camdata = pllab.run_measurements(nloops=nloops, return_data=True)


# # View timings
# plt.figure(1)
# camtimes = pllab.allcams_imtimes[1]
# plt.clf()
# plt.plot(np.diff(camtimes), '-+')


# View camera images
camnum = 1
nshow = 20
startind = 0
clim = [0,1000]

iminds = np.arange(startind, startind+nshow)
camdata = all_camdata[camnum]
plt.figure(2)
for ind in iminds:
    plt.clf()
    plt.imshow(camdata[ind, :, :],clim=clim)
    plt.colorbar()
    plt.title(ind)
    plt.pause(0.1)


# Plot image fluxes
camnum = 0
nshow = 200

plt.figure(4)
plt.clf()
for k in range(2):
    fluxes = pllab.imfluxes(camnum=k)
    plt.subplot(2,1,k+1)
    plt.plot(fluxes[:nshow], '+-')
    plt.title('Mean frame fluxes from camera %d' % k)














