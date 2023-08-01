import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
import matplotlib
matplotlib.use('TkAgg')
plt.ion()

lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'

datadir = '../pllab_data/'
datadir = 'C:/Data/'

slmims_filename = 'slmcube_20230505_zerns_25modes_0.5-0.1_10K_01_file00.npz'
wait_time = 0.1
nloops = 1
showplot = True


if not 'slm' in locals():
    slm = plslm(lutfile=lutfile)
slmimdataf = np.load(datadir + slmims_filename)
slmims = slmimdataf['all_slmims']
slm_flat = np.ones((slmims.shape[1], slmims.shape[2]), dtype='uint8') * 127
n_slmims = slmims.shape[0]
total_nims = nloops * n_slmims

print('Sending SLM images...')
count=0
n_show = n_slmims
n_show = 10
for k in range(nloops):
    for l in range(n_show):
        slmim = slmims[l, :, :]
        slm.slmwrite(slmim, showplot=showplot, skip_readycheck=True)
        # pllab.slm.slmwrite(slmim, showplot=showplot, skip_readycheck=True)
        count += 1
        print(count)
        plt.pause(wait_time)
# slm.slmwrite(slm_flat, showplot=False, skip_readycheck=True)

print('Done.')



