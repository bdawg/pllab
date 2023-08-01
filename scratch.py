plt.figure(1)
plt.clf()

nims = spk.all_pupims.shape[0]
nflx = 27
ampl_params = spk.all_speck_params[:,1,0]

meas_puppower = np.zeros(nims)
meas_psfpower = np.zeros(nims)
meas_fibpower = np.zeros((nims, nflx))
for k in range(nims):
    meas_puppower[k] = np.sum(np.abs(spk.all_pupims[k,:,:])**2)
    meas_psfpower[k] = np.sum(np.abs(spk.all_psfs[k, :, :]) ** 2)
    meas_fibpower[k,:] = np.sum(spk.all_mode_coeffs[k, :27])


plt.clf()
# plt.plot(ampl_params**2)
plt.plot(meas_puppower / np.max(meas_puppower),':')
plt.plot(meas_psfpower / np.max(meas_psfpower), '--')
plt.plot(meas_fibpower / np.max(meas_fibpower))