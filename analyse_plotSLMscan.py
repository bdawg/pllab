import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


datadir = './slm_cals/'
filename = 'slmscanflux_20231023_1024x1024_CALIBRATED-slm6658_at1550_55deg_4pi_20231023_period16_01-10loops.npz'

npf = np.load(datadir+filename)
ampls = npf['ampls']
fluxes = npf['fluxes']

plt.figure(1)
plt.clf()
plt.plot(ampls,fluxes)
# plt.plot(ampls,fluxes_rs, '-o')
plt.xlabel('Stripe amplitude')
plt.ylabel('Zero order flux')
# plt.title('Calibration scan fluxes, period %d px' % period)
plt.title(filename)
plt.tight_layout()
