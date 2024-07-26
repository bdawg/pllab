
import numpy as np
from os.path import splitext
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pllab import pllab
import time

#### Set up required parameters
# datadir = '../pllab_data/'
datadir = 'C:/Data/'
# savedatadir = 'C:/Data/'
savedatadir = datadir
slmdatadir = datadir
slmdatadir = 'Z:/snert/barnaby/PL/202407_labtests/slmcubes/'
# slmdatadir = 'Z:/snert/barnaby/PL/make_probedata/'
darkpath = '../pllab_data/'
# lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_55deg_4pi_20231023.lut'

darkfile = None
# darkfile = 'darks_20240601_laser_1.npz'
darkfile = 'darks_20240708_superK_1.npz' #'darks_20240605_superK_1.npz'


# Choose cube of SLM measurements to perform measurements with
# slmims_filename = 'slmcube_zerns_25modes_05-01_02.npz'
slmims_filename = 'slmcube_20230505_seeing_0.4-10_01.npz'
# slmims_filename = 'slmcube_20230505_seeing_0.4-10_10K_01_file00.npz'
slmims_filename = 'slmcube_20230612_flat127_01.npz'
insert_as_subimage = True

# savefile_prefix = 'pllabdata_20230613_superK'
savefile_prefix = 'pllabdata_20230617d_laser_03'
savefile_prefix = 'pllabdata_20240708_scansrc2_02'

cube_nims = 10000 #10000 # Max number of frames in a cube - cannot be changed without restarting (is used to allocate shm)
camstosave = ['psf','pl'] # camera_index will be assigned in this order
winparams = None # Unused in shm mode cameras (except utility functions like plot_influxes() )
cam_settings = None # Currently unused in shm mode cameras, TODO

camstosave = ['psf','pl'] # 'psf', pl' and 'refl' are valid. Camera_index will be assigned in this order
# cam_tints = [0.00002, 0.00005] # Laser
cam_tints = [0.00005, 0.0005] # SuperK
cam_tints = [0.00007, 0.001] # SuperK, updated for dual-beam
cam_tints = [0.00015, 0.002] # SuperK, more brighter for dual-beam (but don't exceed contr 0.3)


# slmloc=np.array([slm_centre[0], slm_centre[1], slm_rad])
slmloc = None # Use SLM region location from SLM cube file
# slmloc = np.array([565, 390, 190])

# cropdims = None
# cropdims are defined as [FirstColumn, LastColumn, FirstRow, LastRow].
# Columns must be in steps of 32, rows in steps of 4.
# cropdims = [[192, 479, 116, 403], # PSF cam, covers out to 16-pixel-period diffraction
#             [192, 383, 80, 239]] # PL cam, oversized
# cropdims = [[224, 511, 108, 395], # PSF cam, covers out to 16-pixel-period diffraction
#             [192, 383, 68, 227]] # PL cam, oversized
cropdims = [[224, 511, 104, 391], # PSF cam, covers out to 16-pixel-period diffraction
            [192, 383, 60, 219]] # PL cam, oversized
# Offset X,Y, Width, Height: 192, 116, 288, 288 = [192, 479, 116, 403]
# Offset X,Y, Width, Height: 192, 80, 192, 160 = [192, 383, 80, 239]
# Offset X,Y, Width, Height: 224, 108, 288, 288 =
# Offset X,Y, Width, Height: 192, 68, 192, 160 =

savefile_prefix = 'pllabdata_20240717_scansrc2_01'
# savefile_prefix = 'pllabdata_20240603_randsrc2_02'
savefilename = savefile_prefix + '_' + splitext(slmims_filename)[0] + '.npz'

#### Instantiate pllab. This will handle spawning the processes containing plcam instances
pllab = pllab(datadir=datadir, camstosave=camstosave, lutfile=lutfile, winparams=winparams,
              cam_settings=cam_settings, verbose=True, cube_nims=cube_nims, shm_mode=True,
              cropdims=cropdims, darkpath=darkpath, darkfile=darkfile, delays=(48,3))

#### Set camera settings if needed
for k in range(len(camstosave)):
    cmd_str = 'set tint %f' % cam_tints[k]
    pllab.send_shm_camcommand(cam_index=k, cmd_string=cmd_str)

# ## To take darks (make sure light source is off!)
# pllab.take_darks(darkfile=darkfile, save=True)




### Take some measurements
only_first_n = 900
# only_first_n = None

pllab.load_slmims(savedatadir+slmims_filename, insert_as_subimage=insert_as_subimage, slmloc=slmloc,
                  only_first_n=only_first_n)
# pllab.load_slmims(savedatadir+slmims_filename, insert_as_subimage=insert_as_subimage, slmim_array_name='array',
#                   slmim_param_name='info', slmloc=slmloc)

### Move Source2
# xrange = [-1, 1]
# yrange = [-1, 1]
# Roughly centred:
xrange = [-0.6, 0.4]
yrange = [-0.4, 0.6]
contr_range = [0.2, 0.2]
contr_range = [0.3, 0.3]

# Zoomed to PL region
xrange = [-0.3, 0.1]
yrange = [-0.1, 0.3]
# #Updated 20240708:
# xrange = [-0.4, 0.0]
# # yrange = [-0.07, 0.33]
# yrange = [-0.1, 0.3]
# #Updated 20240717:
# xrange = [-0.38, 0.02]
# yrange = [-0.1, 0.3]

# contr_range = [0.3, 1]
nsteps = 30**2
# nreps = 1
savefile_prefix = 'pllabdata_20240717_scansrc2_07'
savefilename = savefile_prefix + '_' + splitext(slmims_filename)[0] + '.npz'

# X,Y scan
xposns = np.linspace(xrange[0], xrange[1], int(np.sqrt(nsteps)))
yposns = np.linspace(yrange[0], yrange[1], int(np.sqrt(nsteps)))
xyposns = []
totsteps = np.shape(xposns)[0] * np.shape(yposns)[0]
for rep in range(1):
    for xposn in xposns:
        for yposn in yposns:
            xyposns.append([xposn, yposn, contr_range[0]])
src2params = np.array(xyposns).T

# ##### Random posn - one off
# xposns = np.random.uniform(xrange[0], xrange[1], nsteps)
# yposns = np.random.uniform(yrange[0], yrange[1], nsteps)
# contrs = np.random.uniform(contr_range[0], contr_range[1], nsteps)
# src2params = np.array([xposns, yposns, contrs])
# src2params_orig = np.copy(src2params)

np.savez(datadir+savefile_prefix+'_src2params.npz', src2params=src2params, xposns=xposns, yposns=yposns)
# all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=1000)
all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=900, src2params=src2params) #900

#Save data
pllab.savedata(filename=savefilename, savedir=savedatadir)



#### Random posn, shuffle same posns
# for k in range(10):
#     src2params = np.copy(src2params_orig)
#     np.random.shuffle(src2params.T)
#
#     # all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=1000)
#     all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=900, src2params=src2params)
#
#     savefilename = savefile_prefix + '_' + '%.3d' % k
#     np.savez(datadir + savefilename + '_src2params.npz', src2params=src2params, xposns=None, yposns=None,
#              contrs=None)
#     ### Save data
#     pllab.savedata(filename=savefilename, savedir=savedatadir)



#### Or take multiple measurements #####################################
num_files = 10
# savefile_prefix = 'pllabdata_20230619a_laser'
# savefile_prefix = 'pllabdata_20230628a_superK'
savefile_prefix = 'pllabdata_20240605_randsrc2_02'
savefile_prefix = 'pllabdata_20240605_singlepsf_01'
savefile_prefix = 'pllabdata_20240708_randsrc2_seeing-rep2_01'
savefile_prefix = 'pllabdata_20240708_randsrc2_seeingcontig-rep1000_01'

# slm_fileprefix = 'slmcube_202300611_zerns25m_0.5-0.1_plusseeing_0.4-10scl0.5_10K_01'
# slm_fileprefix = 'slmcube_20230505_seeing_0.4-10_10K_01'
# slm_fileprefix = 'slmcube_20230505_zerns_25modes_0.4_10K_01'
# slm_fileprefix = 'slmcube_20230505_zerns_25modes_0.5-0.1_10K_01'
slm_fileprefix = 'slmcube_202400708_seeing_0.4-10-scl1_rand-flatn2_10K_01'
slm_fileprefix = 'slmcube_202400708_seeing_0.4-10-scl1_contig-flatn1000_10K_01'

## For src2 stuff:
# Roughly centred:
# xrange = [-0.6, 0.4]
# yrange = [-0.4, 0.6]
contr_range = [0.3, 0.6]
# Zoomed to PL region
# Zoomed to PL region
xrange = [-0.3, 0.1]
yrange = [-0.1, 0.3]
#Updated 20240708:
xrange = [-0.4, 0.0]
# yrange = [-0.07, 0.33]
yrange = [-0.1, 0.3]

src2params_nreps = None
src2params_nreps = 1000 # Repeat each value n times in a row

src2params = None
current_cube_nims = 10000# None
for fnum in range(0, num_files):
    # New slm file for each out file
    cur_slmfilename = slm_fileprefix + '_file%.2d' % fnum + '.npz'
    savefilename = savefile_prefix + '_' + cur_slmfilename

    # # Or reuse same slm file each time
    # cur_slmfilename = slmims_filename  # If using same slmims for each file
    # savefilename = savefile_prefix + '_' + cur_slmfilename[:-4] + '_file%.2d' % fnum + '.npz'

    # If doing src2 things:
    xposns = np.random.uniform(xrange[0], xrange[1], current_cube_nims)
    yposns = np.random.uniform(yrange[0], yrange[1], current_cube_nims)
    contrs = np.random.uniform(contr_range[0], contr_range[1], current_cube_nims)
    if src2params_nreps is not None:
        xposns = np.repeat(xposns, src2params_nreps)[:current_cube_nims]
        yposns = np.repeat(yposns, src2params_nreps)[:current_cube_nims]
        contrs = np.repeat(contrs, src2params_nreps)[:current_cube_nims]
    src2params = np.array([xposns, yposns, contrs])
    np.savez(savedatadir + savefilename[:-4] + '_src2params.npz', src2params=src2params, xposns=xposns, yposns=yposns)

    pllab.load_slmims(slmdatadir + cur_slmfilename, insert_as_subimage=insert_as_subimage, slmloc=slmloc)
    all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=current_cube_nims, src2params=src2params)
    pllab.savedata(filename=savefilename, savedir=savedatadir)



slm_flat = np.ones((1024,1024), dtype='uint8') * 127
pllab.slm.slmwrite(slm_flat, showplot=False)






