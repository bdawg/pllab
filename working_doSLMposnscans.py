import matplotlib.pyplot as plt
import numpy as np
from os.path import splitext
import time
import matplotlib
matplotlib.use('TkAgg')
from pllab import pllab


#### Set up required parameters
datadir = '../pllab_data/'
darkpath = '../pllab_data/'
# lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_55deg_4pi_20231023.lut'

darkfile = 'darks_20240604_superK_1.npz'

cube_nims = 5000 # Max number of frames in a cube - cannot be changed without restarting (is used to allocate shm)
camstosave = ['psf','pl'] # camera_index will be assigned in this order
cam_tints = [0.00002, 0.00005]
cam_tints = [0.00007, 0.001] # SuperK, updated for dual-beam
# cam_tints = [0.00014, 0.002] # SuperK, updated for dual-beam

# winparams_fluxsum = [[257, 340, 42], # Full-frame mode
#                     [160, 282, 120]]
winparams_fluxsum = [[145, 141, 42],  # Crop mode
                     [88, 77, 120]]
winparams_fluxsum = [[138, 138, 42],  # Crop mode
                     [88, 77, 120]]
winparams_fluxsum = [[140, 133, 24],  # Crop mode
                     [88, 77, 120]]

cropdims = [[224, 511, 104, 391], # PSF cam, covers out to 16-pixel-period diffraction
            [192, 383, 60, 219]] # PL cam, oversized

pllab = pllab(datadir=datadir, camstosave=camstosave, lutfile=lutfile,
              verbose=True, cube_nims=cube_nims, shm_mode=True,
              darkpath=darkpath, darkfile=darkfile, winparams_fluxsum=winparams_fluxsum,
              cropdims=cropdims, delays=(48,3))

for k in range(len(camstosave)):
    cmd_str = 'set tint %f' % cam_tints[k]
    pllab.send_shm_camcommand(cam_index=k, cmd_string=cmd_str)

num_lin_posns=32
# meas_range = [350, 750, 175, 575]
meas_range =None
meas_range = [330, 730, 360, 760]
outnamepref = 'slmscanmap_20240605_02'
# meas_range = None
# scanmap = pllab.slmposnscan(num_lin_posns=num_lin_posns, meas_range=meas_range, ampl=72, offset=0)
# np.savez(datadir+outnamepref+'.npz', scanmap=scanmap, meas_range=meas_range,
#          num_lin_posns=num_lin_posns)

### testing - average multiple
nits = 10
all_scanmaps = []
for it in range(nits):
    print('Scan repetition %d of %d'% (it+1, nits))
    scanmap = pllab.slmposnscan(num_lin_posns=num_lin_posns, meas_range=meas_range, ampl=72, offset=0)
    all_scanmaps.append(scanmap)
av_scanmap=np.array(all_scanmaps).mean(0)
plt.imshow(av_scanmap)
np.savez(datadir+outnamepref+'.npz', scanmap=av_scanmap, meas_range=meas_range,
         num_lin_posns=num_lin_posns)

tm = time.time()
np.savez_compressed(datadir+outnamepref+'_slmims.npz', all_slmims=pllab.all_slmims)#,
                    #all_slmim_params=pllab.all_slmim_params)
print('Time to save SLM ims (s): %.1f' % (time.time()-tm))

# Growing-circle scan:
outnamepref = 'slmscanmap_20240605_02d'
circle_centre = [510,566]
num_lin_posns = 400 #64
meas_range = [0, 400, 0, 0]
# scanmap = pllab.slmposnscan(num_lin_posns=num_lin_posns, meas_range=meas_range,
#                             circle_centre=circle_centre, ampl=255)
nits = 2
all_scanmaps = []
for it in range(nits):
    print('Scan repetition %d of %d'% (it+1, nits))
    scanmap = pllab.slmposnscan(num_lin_posns=num_lin_posns, meas_range=meas_range, circle_centre=circle_centre,
                                ampl=72, offset=0)
    all_scanmaps.append(scanmap)
scanmap=np.array(all_scanmaps).mean(0)
np.savez(datadir+outnamepref+'.npz', scanmap=scanmap, meas_range=meas_range,
         num_lin_posns=num_lin_posns)

tm = time.time()
np.savez_compressed(datadir+outnamepref+'_slmims.npz', all_slmims=pllab.all_slmims)#,
                    #all_slmim_params=pllab.all_slmim_params)
print('Time to save SLM ims (s): %.1f' % (time.time()-tm))
