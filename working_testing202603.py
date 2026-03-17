import numpy as np
from os.path import splitext
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pllab import pllab
import time


#### Set up required parameters
datadir = 'C:/Data/'
savedatadir = datadir
slmdatadir = datadir

darkpath = '../pllab_data/'
# darkfile = 'darks_20240822_superK_1.npz'
darkfile = 'darks_20260313a.npz'
darkfile = 'darks_20260313b.npz'
# To take darks use pllab.take_darks(darkfile=darkfile, save=True) (make sure light source is off!)

# lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_55deg_4pi_20231023.lut'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'


## General params
cube_nims = 10000 #10000 # MAX number of frames in a cube - cannot be changed without restarting (is used to allocate shm)
camstosave = ['psf','pl'] # 'psf', pl' and 'refl' are valid. Camera_index will be assigned in this order
cam_tints = [0.00005, 0.0005] # SuperK
cam_tints = [0.0005, 0.010]
cam_tints = np.array([0.00015, 0.002])*5

## For the current measurement set:
current_cube_nims = 100 # Actual no. frames per cube (must be < cube_nims)

## Define SLM region - or None to use SLM region location from SLM cube file
# slmloc=np.array([slm_centre[0], slm_centre[1], slm_rad])
slmloc = None # Use SLM region location from SLM cube file
insert_as_subimage = True # True if slm image file is just the active region (not whole 1024x1024 array)

## Define camera subframe region
## cropdims are defined as [FirstColumn, LastColumn, FirstRow, LastRow].
## e.g. Offset X,Y, Width, Height: 192, 116, 288, 288 = [192, 479, 116, 403]
## Columns must be in steps of 32, rows in steps of 4.
# cropdims = None
# cropdims = [[224, 511, 104, 391], # PSF cam, covers out to 16-pixel-period diffraction
#             # [192, 383, 60, 219]] # PL cam, oversized
#             [192+32, 383+32, 60-40, 219-40]]  # PL cam, oversized
cropdims = [[224-32, 511-32, 104, 391], # PSF cam
            [192+32, 383+32, 60+40, 219+40]] # PL cam

## Instantiate pllab. This will handle spawning the processes containing plcam instances
pllab = pllab(datadir=datadir, camstosave=camstosave, lutfile=lutfile,
              verbose=True, cube_nims=cube_nims, shm_mode=True,
              cropdims=cropdims, darkpath=darkpath, darkfile=darkfile, delays=(48,3))

## Set camera settings if needed
for k in range(len(camstosave)):
    cmd_str = 'set tint %f' % cam_tints[k]
    pllab.send_shm_camcommand(cam_index=k, cmd_string=cmd_str)


#### Make a new set of SLM images, ramping amplitude of sine pattern, and measure them
pllab.makestripecube(n_slmims=100, ampl_range=(0,127), type='sine', savefile='slmims_testout.npz',
                     set_as_current=True)
all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=100, plot_final=True)
# pllab.plot_imfluxes(window=False)

pllab.show_ims(ind=0)
# plt.figure()
# dk = pllab.darkframes[1]
# plt.imshow(all_imdata[1][0,:,:]-dk, vmin=0)

for k in range(100):
    pllab.show_ims(ind=k)
    plt.pause(0.01)