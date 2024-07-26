import numpy as np
from matplotlib.patches import Circle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

# datapath = '../pllab_data/202406/'
datapath = 'C:/Data/'
infilename = 'pllabdata_20240605_scansrc2_02_slmcube_20230612_flat127_01.npz'
src2params_filename = 'pllabdata_20240605_scansrc2_02_src2params.npz' #08

infilename = 'pllabdata_20240717_scansrc2_07_slmcube_20230612_flat127_01.npz'
src2params_filename = 'pllabdata_20240717_scansrc2_07_src2params.npz'

npf = np.load(datapath + src2params_filename, allow_pickle=True)
xposns = npf['xposns']
yposns = npf['yposns']
npf = np.load(datapath + infilename, allow_pickle=True)
darks = npf['darkframes']
imcube_psf = npf['imcube_cam0'] - darks[0]
imcube_pl = npf['imcube_cam1'] - darks[1]
npf.close()
imcube_psf[:,0,:] = 0
imcube_pl[:,0,:] = 0
# plt.imshow(imcube_pl[500,:,:])
subwin = [110, 175, 100, 165]
imcube_psfc = imcube_psf[:, subwin[0]:subwin[1], subwin[2]:subwin[3]]

sz = len(xposns)
pl_totfluxes = np.sum(imcube_pl,axis=(1,2))
pl_fluxmap = pl_totfluxes.reshape(sz,sz)

x = xposns
y = yposns
brightness = pl_totfluxes

# Create a meshgrid for x and y coordinates
# X, Y = np.meshgrid(x, y)
Y, X = np.meshgrid(y, x)

# Reshape brightness to match the shape of X and Y
brightness = brightness.reshape(X.shape)

# Plot the data
plt.figure(1, figsize=(8, 6))
plt.clf()
plt.pcolormesh(X, Y, brightness, shading='auto', cmap='viridis')
plt.colorbar(label='Brightness')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('PL Brightness Distribution')
plt.show()

plt.figure(2)
plt.clf()
plt.imshow(np.mean(imcube_psfc, 0)**0.5)

plt.figure(3)
plt.clf()
psf_maxvals = np.max(imcube_psfc, axis=(1,2))
psf_maxvals = psf_maxvals.reshape(X.shape)
plt.imshow(psf_maxvals.T, extent=[x[0], x[-1] ,y[0], y[-1]])
# plt.axis('auto')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('PSF max val')

# xr [-0.05, 0.24]
# yr [-0.03, -0.3]

# xposns = src2params[0,:]
# yposns = src2params[1,:]
# pl_totfluxes = np.sum(imcube_pl,axis=(1,2))
# sz = int(np.sqrt(src2params.shape[1]))
# pl_fluxmap = pl_totfluxes.reshape(sz,sz)
# plt.imshow(pl_fluxmap)

# pl_fluxmap = np.zeros((sz, sz))
# for x in range(sz):
#     for y in range(sz):
#         pl_fluxmap

# X, Y = np.meshgrid(xposns, yposns)
# brightness = pl_totfluxes.reshape(X.shape)

# x = xposns
# y = yposns
# brightness = pl_totfluxes
# brightness = brightness.reshape((30, 30))
#
# # Plot the data
# plt.figure(figsize=(8, 6))
# plt.pcolormesh(x, y, brightness, shading='auto', cmap='viridis')
# plt.colorbar(label='Brightness')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('Brightness Distribution')
# plt.show()

# # Plot the data using tricontourf
# plt.figure(figsize=(8, 6))
# plt.tricontourf(x, y, brightness, cmap='viridis')
# plt.colorbar(label='Brightness')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('Brightness Distribution')
# plt.show()
#
# # Plot the data using tripcolor
# plt.figure(figsize=(8, 6))
# plt.tripcolor(x, y, brightness, cmap='viridis')
# plt.colorbar(label='Brightness')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('Brightness Distribution')
# plt.show()


