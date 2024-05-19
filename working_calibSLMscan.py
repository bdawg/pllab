
import numpy as np
from os.path import splitext
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pllab import pllab
import time

"""
From Meadowlark docs:
These images can either be stripes or checkerboards with one value held constant (typically 0) and the 
other varying from 0 to the maximum greyscale level. We recommend using stripes if measuring first 
order intensity to ease alignment. The width of the diffraction grating pattern written to the SLM 
during this test should be sufficiently small to clearly separate the 0th and 1st order. We typically 
use, and recommend, either 4 or 8 pixels per stripe.
"""
#### Set up required parameters
# datadir = '../pllab_data/'
datadir = 'C:/Data/'

# savedatadir = 'C:/Data/'
savedatadir = datadir

darkpath = '../pllab_data/'
# lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'
# lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_55deg_4pi_20231023.lut'

# pllab.slm.load_lut('C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_p32ord1.lut')
# pllab.slm.load_lut(lutfile)

darkfile = None
darkfile = 'darks_20230619a_laser_1.npz'
darkfile = 'darks_20230628a_superK_1.npz'
darkfile = 'darks_20231012a_laser_1.npz'
take_new_darks = False

cube_nims = 10000 #10000 # Max number of frames in a cube - cannot be changed without restarting (is used to allocate shm)
camstosave = ['psf','pl'] # camera_index will be assigned in this order
cam_tints = [0.00002, 0.00005] # Laser
# cam_tints = [0.00005, 0.0005] # SuperK
cropdims = [[224, 511, 108, 395], # PSF cam, covers out to 16-pixel-period diffraction
            [192, 383, 68, 227]] # PL cam, oversized

insert_as_subimage = False
slmloc = None # Use SLM region location from SLM cube file
slmloc = np.array([565, 390, 190])
#326,255

enable_cameras = True


####

pllab = pllab(datadir=datadir, camstosave=camstosave, lutfile=lutfile, winparams=None,
              cam_settings=None, verbose=True, cube_nims=cube_nims, shm_mode=True,
              cropdims=cropdims, darkpath=darkpath, darkfile=darkfile, delays=(48,3),
              enable_cameras=enable_cameras)
if enable_cameras:
    for k in range(len(camstosave)):
        cmd_str = 'set tint %f' % cam_tints[k]
        pllab.send_shm_camcommand(cam_index=k, cmd_string=cmd_str)

if take_new_darks:
    ## To take darks (make sure light source is off!)
    pllab.take_darks(darkfile=darkfile, save=True)
else:
    pllab.load_darks()



num_posns = 256
nloops = 10
ampl_range = (0,255)
period=16# 8#32
inverse_stripes = True

slmcube = pllab.makestripecube(num_posns, period, ampl_range, set_as_current=True, showplot=False, return_cube=True)
ampls = pllab.all_slmim_params[:,1]
if inverse_stripes:
    slmcube = -slmcube + 255
    pllab.all_slmims = slmcube



# pllab.show_slm_seq(waittime=0.01, current_cube_nims=num_posns*nloops)
# slm_flat = np.ones((1024,1024), dtype='uint8') * 127
# pllab.slm.slmwrite(slm_flat, showplot=False)

all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=num_posns*nloops)
slm_flat = np.ones((1024,1024), dtype='uint8') * 127
pllab.slm.slmwrite(slm_flat, showplot=False)

raw_ims = all_imdata[0]
imsz = raw_ims.shape[1]
raw_ims = raw_ims.reshape(nloops,num_posns,imsz,imsz)
all_ims = np.mean(raw_ims,0)

# winparams_fluxsum_cam0 = [147, 101, 24] # Zero order
# winparams_fluxsum_cam0 = [147, 159, 24] # 1st order p32
# winparams_fluxsum_cam0 = [147, 255, 24] # 1st order p12

winparams_fluxsum_cam0 = [151, 107, 24] # Zero order 20231016 55deg
# winparams_fluxsum_cam0 = [150, 166, 24] # 1st order p32 20231016 55deg
# winparams_fluxsum_cam0 = [151, 223, 24] # 1st order p16 20231016 55deg

winparams_fluxsum_cam0 = [148, 106, 24] # Zero order 20231023 55deg
# winparams_fluxsum_cam0 = [149, 164, 24] # 1st order p32 20231019 55deg
# winparams_fluxsum_cam0 = [149, 183, 24] # 1st order p24 20231019 55deg
# winparams_fluxsum_cam0 = [149, 221, 24] # 1st order p16 20231019 55deg
# winparams_fluxsum_cam0 = [148, 259, 24] # 1st order p12 20231019 55deg

plt.figure(10)
pllab.winparams_fluxsum[0] = winparams_fluxsum_cam0
pllab.all_imcubes[0][:,:,0] = 0
all_fluxes = pllab.imfluxes(window=True, showplot=True)

fluxes_rs = all_fluxes[0].reshape(nloops,num_posns).T
fluxes = np.mean(fluxes_rs,1)

plt.figure(2)
plt.clf()
plt.plot(ampls,fluxes)
# plt.plot(ampls,fluxes_rs, '-o')
plt.xlabel('Stripe amplitude')
plt.ylabel('Zero order flux')
plt.title('Calibration scan fluxes, period %d px' % period)
plt.tight_layout()


saveoutput=True
if saveoutput:
    savefile = 'slmscanflux_20231023_1024x1024_linearVoltage_from255_55deg_period%d_02-10loops' % period
    # savefile = 'slmscanflux_20231019_1024x1024_linearVoltage_from255_55deg_period%d_1stORDER_01-10loops' % period
    savefile = 'slmscanflux_20231023_1024x1024_CALIBRATED-slm6658_at1550_55deg_4pi_20231023_period%d_01-10loops' % period
    # savefile = 'slmscanflux_20231012_slm6658_at1550_75C_p8_20231012_from0_period%d_02-10loops' % period
    np.savez('./slm_cals/'+ savefile+'.npz', ampls=ampls, fluxes=fluxes, fluxes_rs=fluxes_rs, lutfile=lutfile)
    outarray = np.vstack((ampls, fluxes)).T
    np.savetxt('./slm_cals/'+ savefile+'.csv', outarray, fmt=['%d', '%f'], delimiter=', ')
