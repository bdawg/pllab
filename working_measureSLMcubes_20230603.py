
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
slmdatadir = 'Z:/snert/barnaby/PL/PL_labdata/202306/slmcubes/'
darkpath = '../pllab_data/'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'

darkfile = None
darkfile = 'darks_20230619a_laser_1.npz'
darkfile = 'darks_20230628a_superK_1.npz'


# Choose cube of SLM measurements to perform measurements with
# slmims_filename = 'slmcube_zerns_25modes_05-01_02.npz'
# slmims_filename = 'slmcube_20230505_seeing_0.4-10_01.npz'
slmims_filename = 'slmcube_20230505_seeing_0.4-10_10K_01_file00.npz'
slmims_filename = 'slmcube_20230612_flat127_01.npz'
insert_as_subimage = True

# savefile_prefix = 'pllabdata_20230613_superK'
savefile_prefix = 'pllabdata_20230617d_laser_03'
savefile_prefix = 'pllabdata_test2'

cube_nims = 10000 #10000 # Max number of frames in a cube - cannot be changed without restarting (is used to allocate shm)
camstosave = ['psf','pl'] # camera_index will be assigned in this order
winparams = None # Unused in shm mode cameras (except utility functions like plot_influxes() )
cam_settings = None # Currently unused in shm mode cameras, TODO

camstosave = ['psf','pl'] # 'psf', pl' and 'refl' are valid. Camera_index will be assigned in this order
# cam_tints = [0.00002, 0.00005] # Laser
cam_tints = [0.00005, 0.0005] # SuperK

# slmloc=np.array([slm_centre[0], slm_centre[1], slm_rad])
slmloc = None # Use SLM region location from SLM cube file
slmloc = np.array([565, 390, 190])

# cropdims = None
# cropdims are defined as [FirstColumn, LastColumn, FirstRow, LastRow].
# Columns must be in steps of 32, rows in steps of 4.
# cropdims = [[192, 479, 116, 403], # PSF cam, covers out to 16-pixel-period diffraction
#             [192, 383, 80, 239]] # PL cam, oversized
cropdims = [[224, 511, 108, 395], # PSF cam, covers out to 16-pixel-period diffraction
            [192, 383, 68, 227]] # PL cam, oversized
# Offset X,Y, Width, Height: 192, 116, 288, 288 = [192, 479, 116, 403]
# Offset X,Y, Width, Height: 192, 80, 192, 160 = [192, 383, 80, 239]
# Offset X,Y, Width, Height: 224, 108, 288, 288 =
# Offset X,Y, Width, Height: 192, 68, 192, 160 =


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

# ### Take some measurements
# pllab.load_slmims(savedatadir+slmims_filename, insert_as_subimage=insert_as_subimage, slmloc=slmloc)
# # pllab.load_slmims(savedatadir+slmims_filename, insert_as_subimage=insert_as_subimage, slmim_array_name='array',
# #                   slmim_param_name='info', slmloc=slmloc)
# all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=1000)

#### Save data
# pllab.savedata(filename=savefilename, savedir=savedatadir)


#### Or take multiple measurements
num_files = 20
# savefile_prefix = 'pllabdata_20230619a_laser'
savefile_prefix = 'pllabdata_20230628a_superK'
# slm_fileprefix = 'slmcube_202300611_zerns25m_0.5-0.1_plusseeing_0.4-10scl0.5_10K_01'
# slm_fileprefix = 'slmcube_20230505_seeing_0.4-10_10K_01'
# slm_fileprefix = 'slmcube_20230505_zerns_25modes_0.4_10K_01'
# slm_fileprefix = 'slmcube_20230505_zerns_25modes_0.5-0.1_10K_01'
slm_fileprefix = 'slmcube_20230628_stripes_02'
for fnum in range(num_files):
    cur_slmfilename = slm_fileprefix + '_file%.2d' % fnum + '.npz'
    pllab.load_slmims(slmdatadir + cur_slmfilename, insert_as_subimage=insert_as_subimage, slmloc=slmloc)
    savefilename = savefile_prefix + '_' + cur_slmfilename
    all_imdata = pllab.run_measurements_shm(return_data=True)
    pllab.savedata(filename=savefilename, savedir=savedatadir)

# num_files = 10
# savefile_prefix = 'pllabdata_20230618a_laser'
# # savefile_prefix = 'pllabdata_20230616_superK'
# # slm_fileprefix = 'slmcube_202300611_zerns25m_0.5-0.1_plusseeing_0.4-10scl0.5_10K_01'
# # slm_fileprefix = 'slmcube_20230505_seeing_0.4-10_10K_01'
# # slm_fileprefix = 'slmcube_20230505_zerns_25modes_0.4_10K_01'
# # slm_fileprefix = 'slmcube_20230505_zerns_25modes_0.5-0.1_10K_01'
# slm_fileprefix = 'slmcube_20230617_complsines-10sp_01'
# for fnum in range(num_files):
#     cur_slmfilename = slm_fileprefix + '_file%.2d' % fnum + '.npz'
#     pllab.load_slmims(datadir + cur_slmfilename, insert_as_subimage=insert_as_subimage, slmloc=slmloc)
#     savefilename = savefile_prefix + '_' + cur_slmfilename
#     all_imdata = pllab.run_measurements_shm(return_data=True)
#     pllab.savedata(filename=savefilename, savedir=savedatadir)


# num_files = 10
# savefile_prefix = 'pllabdata_20230617e_superK'
# # savefile_prefix = 'pllabdata_20230616_superK'
# # slm_fileprefix = 'slmcube_202300611_zerns25m_0.5-0.1_plusseeing_0.4-10scl0.5_10K_01'
# # slm_fileprefix = 'slmcube_20230505_seeing_0.4-10_10K_01'
# slm_fileprefix = 'slmcube_20230505_zerns_25modes_0.4_10K_01'
# # slm_fileprefix = 'slmcube_20230505_zerns_25modes_0.5-0.1_10K_01'
# for fnum in range(num_files):
#     cur_slmfilename = slm_fileprefix + '_file%.2d' % fnum + '.npz'
#     pllab.load_slmims(datadir + cur_slmfilename, insert_as_subimage=insert_as_subimage, slmloc=slmloc)
#     savefilename = savefile_prefix + '_' + cur_slmfilename
#     all_imdata = pllab.run_measurements_shm(return_data=True)
#     pllab.savedata(filename=savefilename, savedir=savedatadir)
#
# num_files = 10
# savefile_prefix = 'pllabdata_20230617e_superK'
# # savefile_prefix = 'pllabdata_20230616_superK'
# # slm_fileprefix = 'slmcube_202300611_zerns25m_0.5-0.1_plusseeing_0.4-10scl0.5_10K_01'
# slm_fileprefix = 'slmcube_20230505_seeing_0.4-10_10K_01'
# # slm_fileprefix = 'slmcube_20230505_zerns_25modes_0.4_10K_01'
# # slm_fileprefix = 'slmcube_20230505_zerns_25modes_0.5-0.1_10K_01'
# for fnum in range(num_files):
#     cur_slmfilename = slm_fileprefix + '_file%.2d' % fnum + '.npz'
#     pllab.load_slmims(datadir + cur_slmfilename, insert_as_subimage=insert_as_subimage, slmloc=slmloc)
#     savefilename = savefile_prefix + '_' + cur_slmfilename
#     all_imdata = pllab.run_measurements_shm(return_data=True)
#     pllab.savedata(filename=savefilename, savedir=savedatadir)




slm_flat = np.ones((1024,1024), dtype='uint8') * 127
pllab.slm.slmwrite(slm_flat, showplot=False)
#
# # pllab.slm.makeramp(xslope=0.2, yslope=0, showplot=True, sendtoslm=True)
# for k in np.linspace(-1,1,25):
#     pllab.slm.makeramp(xslope=0, yslope=k, showplot=True, sendtoslm=True)
#     plt.pause(0.001)





