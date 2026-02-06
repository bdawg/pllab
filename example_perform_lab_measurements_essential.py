import numpy as np
from os.path import splitext
import matplotlib.pyplot as plt
from pllab import pllab
import time


#### Set up required parameters
datadir = 'C:/Data/'
savedatadir = datadir
slmdatadir = datadir

darkpath = '../pllab_data/'
darkfile = 'darks_20240822_superK_1.npz'
# To take darks use pllab.take_darks(darkfile=darkfile, save=True) (make sure light source is off!)

lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_55deg_4pi_20231023.lut'

## Choose cube of SLM measurements to perform measurements with
slmims_filename = 'slmcube_20230505_seeing_0.4-10_01.npz'

## Set output file prefix
savefile_prefix = 'pllabdata_20240708_scansrc2_02'

## General params
cube_nims = 10000 #10000 # MAX number of frames in a cube - cannot be changed without restarting (is used to allocate shm)
camstosave = ['psf','pl'] # 'psf', pl' and 'refl' are valid. Camera_index will be assigned in this order
cam_tints = [0.00005, 0.0005] # SuperK

## For the current measurement set:
current_cube_nims = 10000 # Actual no. frames per cube (must be < cube_nims)

## Define SLM region - or None to use SLM region location from SLM cube file
# slmloc=np.array([slm_centre[0], slm_centre[1], slm_rad])
slmloc = None # Use SLM region location from SLM cube file
insert_as_subimage = True # True if slm image file is just the active region (not whole 1024x1024 array)

## Define camera subframe region
## cropdims are defined as [FirstColumn, LastColumn, FirstRow, LastRow].
## e.g. Offset X,Y, Width, Height: 192, 116, 288, 288 = [192, 479, 116, 403]
## Columns must be in steps of 32, rows in steps of 4.
# cropdims = None
cropdims = [[224, 511, 104, 391], # PSF cam, covers out to 16-pixel-period diffraction
            [192, 383, 60, 219]] # PL cam, oversized


## Set the locations / contrast for source 2 (the 'planet')
## Format is src2params = np.array([xposns, yposns, contrs])
# E.g. for random positions within a specified range:
xrange = [-0.6, 0.4]
yrange = [-0.4, 0.6]
contr_range = [0.3, 1]
xposns = np.random.uniform(xrange[0], xrange[1], current_cube_nims)
yposns = np.random.uniform(yrange[0], yrange[1], current_cube_nims)
contrs = np.random.uniform(contr_range[0], contr_range[1], current_cube_nims)
src2params = np.array([xposns, yposns, contrs])




#### Do the measurements

## Instantiate pllab. This will handle spawning the processes containing plcam instances
pllab = pllab(datadir=datadir, camstosave=camstosave, lutfile=lutfile,
              verbose=True, cube_nims=cube_nims, shm_mode=True,
              cropdims=cropdims, darkpath=darkpath, darkfile=darkfile, delays=(48,3))

## Set camera settings if needed
for k in range(len(camstosave)):
    cmd_str = 'set tint %f' % cam_tints[k]
    pllab.send_shm_camcommand(cam_index=k, cmd_string=cmd_str)

## Load the slm patterns
pllab.load_slmims(savedatadir+slmims_filename, insert_as_subimage=insert_as_subimage, slmloc=slmloc)

## Actually run the measurements
all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=current_cube_nims, src2params=src2params)

## Save results (and source2 positions)
savefilename = savefile_prefix + '_' + splitext(slmims_filename)[0] + '.npz'
pllab.savedata(filename=savefilename, savedir=savedatadir)
np.savez(savedatadir + savefilename[:-4] + '_src2params.npz', src2params=src2params, xposns=xposns, yposns=yposns)


# Set the SLM back to a nominal flat
slm_flat = np.ones((1024,1024), dtype='uint8') * 127
pllab.slm.slmwrite(slm_flat, showplot=False)

