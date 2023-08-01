import numpy as np


datadir = 'C:/Data/'
# datadir = '/Users/bnorris/DontBackup/PL/202306/'

slm_fileprefix = 'pllabdata_20230625a_superK_slmcube_20230625_complsines-01sp_04'
# slm_fileprefix = 'pllabdata_20230625a_superK_slmcube_20230625_complsines-01sp_06'
savefile_prefix = 'slmcoeffs'
num_files = 10
skipfile = None
skipfile = 3
remove_lastmode = True

combined_slmimparams_list = []
combined_coeffs_list = []
fnums = np.arange(num_files)
if skipfile is not None:
    fnums = fnums[fnums != skipfile]
    num_files -= 1

for fnum in fnums:
    cur_slmfilename = slm_fileprefix + '_file%.2d' % fnum + '.npz'
    slmimdataf = np.load(datadir + cur_slmfilename, allow_pickle=True)
    cur_slmim_params = slmimdataf['all_slmim_params'].item()
    combined_slmimparams_list.append(cur_slmim_params)
    combined_coeffs_list.append(cur_slmim_params['coeffs'])

combined_coeffs = np.array(combined_coeffs_list)
# nmodes = combined_coeffs.shape[2]
# combined_coeffs = np.squeeze(combined_coeffs.reshape((1,-1,nmodes)))

nfrms = combined_coeffs.shape[1]
nmodes = combined_coeffs.shape[2]
if remove_lastmode:
    nmodes = nmodes-1
nspecks = combined_coeffs.shape[3]
npars = nmodes*nspecks
combined_coeffs_out = np.zeros((num_files*nfrms, npars))
for k in range(num_files):
    cur_coeffs = np.zeros((nfrms, nmodes * nspecks))
    for l in range(nspecks):
        cur_coeffs[:, l*nmodes:(l+1)*nmodes] = combined_coeffs[k,:,:nmodes,l]
    combined_coeffs_out[k*nfrms:(k+1)*nfrms, :] = cur_coeffs
combined_coeffs = combined_coeffs_out

savefilename = savefile_prefix + '_' + slm_fileprefix
np.savez(datadir+savefilename+'.npz', combined_coeffs=combined_coeffs,
         combined_slmimparams_list=combined_slmimparams_list,
         ogfilepref=slm_fileprefix)

