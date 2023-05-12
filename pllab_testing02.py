import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from pllab import pllab


datadir = '../pllab_data/'
slmims_filename = 'slmcube_varyingstripes_0-60_10frm_01.npz'
# slmims_filename = 'slmcube_alternatingstripes_0-60_10frm_01.npz'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'

cube_nims = 200
camstosave = ['psf','pl'] # camera_index will be assigned in this order
winparams=None
cam_settings = None

pllab = pllab(datadir=datadir, camstosave=camstosave, lutfile=lutfile, winparams=winparams,
              cam_settings=cam_settings, verbose=True, cube_nims=cube_nims)
pllab.load_slmims(datadir+slmims_filename)
all_imdata = pllab.run_measurements_shm(return_data=True)
# pllab.run_measurements_shm(return_data=False)

winparams_fluxsum = [[257, 340, 42],
                    [160, 282, 120]]
pllab.plot_imfluxes(window=True, winparams=winparams_fluxsum)


