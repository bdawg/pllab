import matplotlib.pyplot as plt
import numpy as np
from os.path import splitext
import matplotlib
matplotlib.use('TkAgg')
from pllab import pllab


#### Set up required parameters
datadir = '../pllab_data/'
darkpath = '../pllab_data/'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'

# darkfile = None
darkfile = 'default_dark_01.npz'


# Choose cube of SLM measurements to perform measurements with
slmims_filename = 'CheckerboardStackTestFlippedPositioned0topi.npz'
slmim_array_name = 'arr_0'

savefile_prefix = 'pllabdata_20230526'


cube_nims = 1000 # Max number of frames in a cube - cannot be changed without restarting (is used to allocate shm)
camstosave = ['psf','pl'] # camera_index will be assigned in this order
winparams = None # Unused in shm mode cameras (except utility functions like plot_influxes() )
cam_settings = None # Currently unused in shm mode cameras, TODO

camstosave = ['psf','pl'] # 'psf', pl' and 'refl' are valid. Camera_index will be assigned in this order
cam_tints = [0.00005, 0.0005]

cropdims = None
# # cropdims are defined as [FirstColumn, LastColumn, FirstRow, LastRow].
# # Columns must be in steps of 32, rows in steps of 4.
# cropdims = [[192, 479, 116, 403], # PSF cam, covers out to 16-pixel-period diffraction
#             [192, 383, 80, 239]] # PL cam, oversized
# # Offset X,Y, Width, Height: 192, 116, 288, 288 = [192, 479, 116, 403]
# # Offset X,Y, Width, Height: 192, 80, 192, 160 = [192, 383, 80, 239]

savefilename = savefile_prefix + '_' + splitext(slmims_filename)[0] + '.npz'

#### Instantiate pllab. This will handle spawning the processes containing plcam instances
pllab = pllab(datadir=datadir, camstosave=camstosave, lutfile=lutfile, winparams=winparams,
              cam_settings=cam_settings, verbose=True, cube_nims=cube_nims, shm_mode=True,
              cropdims=cropdims, darkpath=darkpath)
pllab.load_slmims(datadir+slmims_filename, slmim_array_name=slmim_array_name)

#### Set camera settings if needed
for k in range(len(camstosave)):
    cmd_str = 'set tint %f' % cam_tints[k]
    pllab.send_shm_camcommand(cam_index=k, cmd_string='cmd_str')

#### Take some measurements
all_imdata = pllab.run_measurements_shm(return_data=True)

#### Save data
pllab.savedata(filename=savefilename)








