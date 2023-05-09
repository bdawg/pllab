import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')
plt.ion()


def goodtimer(time_ms):
    tm = time_ms / 1000
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < tm:
        pass


datadir = '../pllab_data/'
slmims_filename = 'slmcube_varyingstripes_0-60_10frm_01.npz'
# slmims_filename = 'slmcube_alternatingstripes_0-60_10frm_01.npz'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'
darkfiledir = '../pllab_data/darks/'
darkfile1 = None #'dark_psfcam_50us'

verbose = True
nloops = 3


cam_syncdelay_ms = 12
extra_delay_ms  = 3
wait_time_ms = cam_syncdelay_ms + extra_delay_ms

if not 'slm' in locals():
    slm = plslm(lutfile=lutfile)


slmimdataf = np.load(datadir + slmims_filename)
slmims = slmimdataf['all_slmims']
if type(slmims[0, 0, 0]) is not np.int8:
    print('Error: input SLM cube not int8')
all_slmim_params = slmimdataf['all_slmim_params']


n_slmims = slmims.shape[0]

# croppeddark = cam.dark[win[0]:win[1], win[2]:win[3]]
total_nims = nloops * n_slmims

# slm.slmwrite(slmims[0, :, :], showplot=False)
time.sleep(0.1)



print('Starting data acquisition')
startime = time.time()

count=0
for k in range(nloops):
    for l in range(n_slmims):
        slmim = slmims[l, :, :]
        slm.slmwrite(slmim, showplot=False, skip_readycheck=True)
        count += 1

        goodtimer(wait_time_ms)

print('Done - elapsed time %.2f seconds' % (time.time() - startime))




