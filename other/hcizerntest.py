import hcipy
from scipy.ndimage.measurements import center_of_mass as com
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pickle


n_px = 128
nmodes = 1000

# outfilename = 'zernikecube_n100_512px'
outfilename = 'zernikecube_n1000_128px'

D_tel = 1 # meter
pupil_grid = hcipy.make_pupil_grid(n_px, D_tel)

# focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, 8, 16*10, wavelength=wavelength)
# prop = FraunhoferPropagator(pupil_grid, focal_grid)
# aperture = circular_aperture(D_tel)(pupil_grid)
# wf = Wavefront(aperture, wavelength)

zern_basis = hcipy.mode_basis.make_zernike_basis(num_modes=nmodes, D=D_tel, grid=pupil_grid)

zerncube = np.zeros((n_px, n_px, nmodes))
for k in range(nmodes):
    wf = np.array(zern_basis[k].reshape(n_px,n_px))
    zerncube[:,:,k] = wf

    # plt.clf()
    # plt.imshow(wf)
    # plt.pause(0.1)

# np.save(outfilename+'.npy', zerncube)


# # Check normalisation
# gY, gX = np.ogrid[:n_px, :n_px]
# cnt = int(n_px/2)
# dist = np.sqrt( (gX - cnt)**2 + (gY - cnt)**2)
# mask = dist >= n_px/2
# wf_masked = np.ma.masked_array(wf, mask)
# plt.imshow(wf_masked)
# print(np.std(wf_masked))















# ####
# #Magic modes...
# # aka DM Karhunen–Loève modes?
#
# n_px = 64
# gY, gX = np.mgrid[:n_px, :n_px]
# cnt = int(n_px/2)
# dist = np.sqrt( (gX - cnt)**2 + (gY - cnt)**2)
# mask = dist >= n_px/2
#
# X=gX[~mask]
# Y=gY[~mask]
# d = ((X[None,:]-X[:,None])**2 + (Y[None,:]-Y[:,None])**2)**0.66
# T, U, V = np.linalg.svd(d)
# m = np.zeros((3205, 64, 64))
# for k in range(3205):
#     m[k, ~mask] = V[:,k]
#
# for k in range(1000):
#     plt.clf()
#     plt.imshow(m[k,:,:])
#     plt.pause(0.01)
#
# for k in range(3205):
#     plt.clf()
#     plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(m[k,:,:])))), cmap='magma', vmin=-3, vmax=3)
#     plt.pause(0.001)