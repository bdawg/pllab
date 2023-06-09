import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')


datadir = '../pllab_data/'
slm = plslm()



# Make set of alternating, random stripes
n_slmims = 1000
period_range = [5,200] #[60,60]
ampl_range = [20,100]

outname = 'slmcube_alternatingstripes_0-60_10frm_01.npz'
n_slmims = 10
period_range = [50,50] #[60,60]
ampl_range = [60,60]

slmim_dtype = 'int8'
all_slmims = np.zeros((n_slmims, slm.slmdims[0], slm.slmdims[1]), dtype=slmim_dtype)
all_slmim_params = np.zeros((n_slmims, 2))
for k in range(n_slmims):
    if k % 2 == 0:
        all_slmims[k, :, :] = np.zeros((slm.slmdims[0], slm.slmdims[1]))
    else:
        period = np.random.uniform(period_range[0], period_range[1])
        amplitude = np.random.uniform(ampl_range[0], ampl_range[1])
        slm.makestripes(period=period, ampl=amplitude, phi=0, type='square', showplot=False, sendtoslm=False)
        slm_im = slm.nextim.astype(slmim_dtype)
        all_slmims[k, :, :] = slm_im
        all_slmim_params[k, :] = [period, amplitude]



# Vary stripe amplitude
outname = 'slmcube_varyingstripes_0-60_4frm_01.npz'
n_slmims = 4
ampl_range = [0,60]
period_range = [50]

amplvals = np.linspace(ampl_range[0], ampl_range[1], n_slmims)
slmim_dtype = 'int8'
all_slmims = np.zeros((n_slmims, slm.slmdims[0], slm.slmdims[1]), dtype=slmim_dtype)
all_slmim_params = np.zeros((n_slmims, 2))
for k in range(n_slmims):
        period = period_range[0]
        amplitude = amplvals[k]
        slm.makestripes(period=period, ampl=amplitude, phi=0, type='square', showplot=False, sendtoslm=False)
        slm_im = slm.nextim.astype(slmim_dtype)
        all_slmims[k, :, :] = slm_im
        all_slmim_params[k, :] = [period, amplitude]

tm = time.time()
np.savez_compressed(datadir+outname, all_slmims=all_slmims,
                    all_slmim_params=all_slmim_params)
print('Time to save SLM ims (s): %.1f' % (time.time()-tm))


