import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from pllab import pllab


#### Set up required parameters
datadir = '../pllab_data/'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'

# Choose cube of SLM measurements to perform measurements with
slmims_filename = 'slmcube_varyingstripes_0-60_10frm_01.npz'
slmims_filename = 'slmcube_alternatingstripes_0-60_10frm_01.npz'

cube_nims = 100#0 # Max number of frames in a cube - cannot be changed without restarting (is used to allocate shm)
camstosave = ['psf','pl'] # camera_index will be assigned in this order
winparams=None # Unused in shm mode cameras (except utility functions like plot_influxes() )
cam_settings = None # Currently unused in shm mode cameras, TODO

camstosave = ['psf','pl'] # 'psf', pl' and 'refl' are valid. Camera_index will be assigned in this order


#### Instantiate pllab. This will handle spawning the processes containing plcam instances
pllab = pllab(datadir=datadir, camstosave=camstosave, lutfile=lutfile, winparams=winparams,
              cam_settings=cam_settings, verbose=True, cube_nims=cube_nims, shm_mode=True)
pllab.load_slmims(datadir+slmims_filename)

#### Try sending a command to camera 1
response = pllab.send_shm_camcommand(cam_index=1, cmd_string='set tint 0.0005', return_response=True)

#### Take some measurements
all_imdata = pllab.run_measurements_shm(return_data=True)

#### Take a quick look at summed fluxes (within some window)
winparams_fluxsum = [[257, 340, 42],
                    [160, 282, 120]]
pllab.plot_imfluxes(window=True, winparams=winparams_fluxsum)


# #### Make a new set of SLM images, ramping amplitude of sine pattern, and measure them
# pllab.makestripecube(n_slmims=100, ampl_range=(0,127), type='sine', savefile='slmims_testout.npz',
#                      set_as_current=True)
# all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=100)
# pllab.plot_imfluxes(window=True, winparams=winparams_fluxsum)
