import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')


n_imstolog = 1000

cam0 = credcam(camera_index=1, verbose=True)
cam1 = credcam(camera_index=1, verbose=True)

cam0.set_nims_tolog(n_imstolog)
cam1.set_nims_tolog(n_imstolog)

cam0.get_n_images(blocking=False)
cam1.get_n_images(blocking=False)
while cam1.nims_lefttolog > 0:
    time.sleep(0.1)
camims0 = cam0.loggedims_cube
camims1 = cam1.loggedims_cube


plt.clf()
plt.subplot(211)
plt.plot(np.diff(cam0.loggedims_times_arr), '-+')
plt.subplot(212)
plt.plot(np.diff(cam1.loggedims_times_arr), '-+')

diffims = np.diff(camims0, axis=0)
fluxsqdifims = np.sum(diffims**2, (1, 2))
ndiffimzeros = np.sum(fluxsqdifims == 0)
print('Number of diffim_sums=0s: %d' % ndiffimzeros)
diffims = np.diff(camims1, axis=0)
fluxsqdifims = np.sum(diffims**2, (1, 2))
ndiffimzeros = np.sum(fluxsqdifims == 0)
print('Number of diffim_sums=0s: %d' % ndiffimzeros)
