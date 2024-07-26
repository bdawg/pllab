import numpy as np
from matplotlib.patches import Circle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

# datapath = '../pllab_data/202406/'
datapath = 'C:/Data/'
infilename = 'pllabdata_20240603_randsrc2_02_001.npz'
src2params_filename = 'pllabdata_20240603_randsrc2_02_001_src2params.npz'


npf = np.load(datapath + src2params_filename, allow_pickle=True)
src2params = npf['src2params']
src2params_og = np.copy(src2params)
npf = np.load(datapath + infilename, allow_pickle=True)
darks = npf['darkframes']
imcube_psf = npf['imcube_cam0'] - darks[0]
imcube_pl = npf['imcube_cam1'] - darks[1]
npf.close()
imcube_psf[:,0,:] = 0
imcube_pl[:,0,:] = 0
# plt.imshow(imcube_pl[500,:,:])

sorted_inds = np.argsort(src2params_og[0, :])
src2params = src2params_og[:, sorted_inds]



subwin = [110, 175, 100, 165]
imcube_psfc = imcube_psf[:, subwin[0]:subwin[1], subwin[2]:subwin[3]]
imcube_psfc = imcube_psfc[sorted_inds, :, :]
plt.figure(2)
plt.clf()
plt.imshow(np.mean(imcube_psfc, 0)**0.5)

k=300
plt.imshow(imcube_psfc_0[k,:,:])

# for k in range(imcube_psfc.shape[0]):
#     plt.clf()
#     plt.imshow(imcube_psfc[k,:,:] ** 0.5)
#     plt.pause(0.2)


xposns = src2params[0,:]
yposns = src2params[1,:]
pl_totfluxes = np.sum(imcube_pl,axis=(1,2))

x = xposns
y = yposns
brightness = pl_totfluxes

# # Plot the data using tricontourf
# plt.figure(figsize=(8, 6))
# plt.tricontourf(x, y, brightness, cmap='viridis')
# plt.colorbar(label='Brightness')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('Brightness Distribution')
# plt.show()

# Plot the data using tripcolor
plt.figure(figsize=(8, 6))
plt.tripcolor(x, y, brightness, cmap='viridis', shading='gouraud')
plt.colorbar(label='Brightness')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Brightness Distribution')
plt.show()


