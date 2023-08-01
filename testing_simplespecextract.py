import numpy as np
import matplotlib.pyplot as plt
import time
plt.ion()
from scipy.optimize import minimize
from matplotlib.patches import Rectangle


datadir = '/Users/bnorris/DontBackup/PL/202306/'
coreposns_dir = '../pllab_data/'
datafilename = 'pllabdata_20230605_superK_slmcube_20230505_seeing_0.4-10_10K_01_file00.npz'
coreposns_file = 'plcoords_mono_20230605_02.npz'

npf = np.load(datadir + datafilename, allow_pickle=True)
imcube_pl = npf['imcube_cam1']
darkframes = npf['darkframes']
imcube_pl = imcube_pl - darkframes[1]
imcube_pl[:, 0, :] = 0
refim = np.mean(imcube_pl, 0)
# refim = np.sqrt(refim)
# refim[np.isnan(refim)] = 0
refim /= np.max(refim)
npf = np.load(coreposns_dir + coreposns_file)
yposn = npf['yposn']
xposn = npf['xposn']
core_posns = np.array([yposn, xposn])
refim_mono = npf['refim']



spec_length = 24
spec_width = 3
callaser_posn = 10

plt.figure(1)
plt.clf()
n_elems = xposn.size
plt.imshow(refim+refim_mono)
plt.plot(xposn, yposn, 'xw')
for k in range(n_elems):
    plt.text(xposn[k], yposn[k], '%d' % (k), color='white')
    plt.gca().add_patch(Rectangle((xposn[k]-callaser_posn, yposn[k]-(spec_width/2)), spec_length, spec_width, edgecolor='white',
                                  fill=False))


im = refim+refim_mono
hw = spec_width // 2
plt.figure(1)
for ind in range(core_posns.shape[1]):
    ps = core_posns[:,ind]
    ps = np.round(ps).astype('int')
    subim = im[ps[0]-hw:ps[0]+hw+1,
            ps[1]-callaser_posn:ps[1]+(spec_length-callaser_posn)]
    plt.clf()
    plt.imshow(subim)
    plt.title(ind)
    plt.pause(0.5)




