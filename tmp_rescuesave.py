# savedir = '/media/data/bnorris/pl_simdata/'
savedir = './'
savefile = 'siminjout_2ssp_ampl01-1_20230728b-01e_%.2d.npz' % 0

psf_cropsize = 32
pupil_rescale = 0.2
pupil_cropsize = 64
slmim_cropsize = 270
slmpsf_cropsize = psf_cropsize

print('Cropping PSF')
psfs = spk.all_psfs
cnt = psfs.shape[1] // 2
hw = psf_cropsize // 2
psfs = psfs[:, cnt - hw:cnt + hw, cnt - hw:cnt + hw]
# psfs = np.hstack((np.real(psfs), np.imag(psfs)))
all_psfs_out = psfs

print('Scaling & cropping pupil images...')
pupims = spk.all_pupims
ndata = pupims.shape[0]
newsz = int(pupims.shape[1] * pupil_rescale)
pupims_rs_rl = np.zeros((ndata, newsz, newsz), dtype='float32')
pupims_rs_im = np.zeros((ndata, newsz, newsz), dtype='float32')
for k in range(ndata):
    if k % 500 == 0:
        print('Resizing pupil im %d' % k)
    pupims_rs_rl[k, :, :] = rescale(pupims.real[k, :, :], pupil_rescale)
    pupims_rs_im[k, :, :] = rescale(pupims.imag[k, :, :], pupil_rescale)
pupims_rs = pupims_rs_rl + 1j * pupims_rs_im
cnt = pupims_rs.shape[1] // 2
hw = pupil_cropsize // 2
pupims_rs = pupims_rs[:, cnt - hw:cnt + hw, cnt - hw:cnt + hw]
# pupims_rs = np.hstack((np.real(pupims_rs), np.imag(pupims_rs)))
all_pupils_out = pupims_rs
del pupims, pupims_rs_rl, pupims_rs_im

# # Cropping SLM im
# print('Cropping SLM im')
# all_slmims_out = np.angle(spk.all_slmims).astype('float32')
# cnt = all_slmims_out.shape[1] // 2
# hw = slmim_cropsize // 2
# all_slmims_out = all_slmims_out[:, cnt - hw:cnt + hw, cnt - hw:cnt + hw]
all_slmims_out = []

# print('Cropping SLM PSF')
# slmpsfs = spk.all_slm_psfs
# cnt = slmpsfs.shape[1] // 2
# hw = slmpsf_cropsize // 2
# slmpsfs = slmpsfs[:, cnt - hw:cnt + hw, cnt - hw:cnt + hw]
# # psfs = np.hstack((np.real(psfs), np.imag(psfs)))
# all_slm_psfs_out = slmpsfs
all_slm_psfs_out = []


print('Saving to ' + savedir + savefile)
np.savez(savedir + savefile, all_speck_params=spk.all_speck_params, all_pupims=all_pupils_out,
         all_psfs=all_psfs_out, all_mode_coeffs=spk.all_mode_coeffs, all_slmims=all_slmims_out,
         all_slm_psfs=all_slm_psfs_out)
# np.savez_compressed(savedir + savefile, all_speck_params=self.all_speck_params, all_pupims=self.all_pupims,
#          all_psfs=self.all_psfs, all_mode_coeffs=self.all_mode_coeffs, all_slmims=self.all_slmims)
print('Saving done.')