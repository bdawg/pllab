import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from pldata import PLdata




# datadir = '/Users/bnorris/DontBackup/PL/202306/'
datadir = 'C:/Data/'
# datadir = '/Volumes/ExtSSD/PL_labdata/Set_20230617e/'
# datadir = '/Volumes/ExtSSD/PL_labdata/Set_20230618a/'
coreposns_dir = '../pllab_data/fluxes/'

# datafilename = 'pllabdata_20230605_slmcube_20230505_zerns_25modes_0.5-0.1_10K_01_file00.npz'
datafilename = 'pllabdata_20230625a_superK_slmcube_20230625_complsines-01sp_03_file00.npz'
# datafilename = 'pllabdata_20230620a_superK_slmcube_20230612_flat127_01.npz'

# Instantiate PLdata
pldata = PLdata(datadir)

# Load some data (used as reference image)
# pldata.load_pldata_file(datafilename)
#
# # Find monochromatic core positions
# coreposns_savefile = 'plcoords_mono_20230617_01.npz'

# # Guesses:
# # edge_length = 13
# # offset = [79, 89]
# # aspect = 0.75
# # rotate = 14
# edge_length = 13
# offset = [77, 108]
# aspect = 0.75
# rotate = 14
#
# p0 = np.array([edge_length, rotate, offset[0], offset[1], aspect])
# pldata.find_core_locations_mono(p0, show_p0_only=True)
#
# pldata.find_core_locations_mono(p0, coreposns_file=coreposns_savefile, savedir=coreposns_dir,
#                                 save=True)
# print(pldata.optimize_result)


# Or load core positions
# coreposns_file = 'plcoords_mono_20230605_02.npz'
coreposns_file = 'plcoords_mono_20230617_01.npz'
pldata.load_coreposns(coreposns_file, loaddir=coreposns_dir)


# Extract fluxes
# pldata.extract_fluxes_mono(nfrms=None, showplot=True)
# pldata.extract_fluxes_poly_simple(nfrms=1, show_main_plot=True, show_indiv_plots=True, spec_width=3)


# # Look at fluxes
# fluxes = pldata.multifile_fluxes
# # fluxes = pldata.all_fluxes
# plt.clf()
# # plt.imshow(fluxes, aspect='auto', interpolation='None')
# plt.imshow(fluxes[9900:10100,:], aspect='auto', interpolation='None')
# # plt.imshow(np.log10(pldata.all_fluxes), aspect='auto', interpolation='None')
# for k in range(55):
#     plt.clf()
#     plt.plot(pldata.all_vprofs[0,k,:])
#     plt.title(k)
#     plt.pause(0.01)


# # Extract multiple files
# filepref = 'pllabdata_20230617a_laser_slmcube_20230505_seeing_0.4-10_10K_01_file'
# pldata.extract_multifile_fluxes(filepref, mode='mono', numfiles=10, savefilename='plfluxes')
filepref = 'pllabdata_20230628a_superK_slmcube_20230628_stripes_02_file'
pldata.extract_multifile_fluxes(filepref, mode='poly', numfiles=20, savefilename='plfluxes_simplepoly_5kpf')#, skipfile=3)
