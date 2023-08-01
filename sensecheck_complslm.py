import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale
plt.ion()

# npx_pup = 1024
# pupildiam_px = 512
# padding_px = 256
# cropsize = 64 # For plotting

npx_pup = 1024//2
pupildiam_px = 512//2
padding_px = 256//2
cropsize = 96 # For plotting
simpsf_cropsize = None
simpsf_cropsize = cropsize

def make_virt_pupil(dc_power, cycles_per_pupil, ampl1, ampl2, rot, phi=0):
    # Make virtual complex pupil
    period = pupildiam_px / cycles_per_pupil
    Y, X = np.mgrid[-npx_pup / 2:npx_pup / 2, -npx_pup / 2:npx_pup / 2]
    R = np.sqrt(X ** 2 + Y ** 2)
    mask = np.zeros((npx_pup, npx_pup), dtype='bool')
    mask[R < pupildiam_px//2] = True
    X = np.roll(X, (0, phi))
    Xr = np.cos(rot / 180 * np.pi) * X + np.sin(rot / 180 * np.pi) * Y
    # Xr = X
    im_c = ampl1 * np.exp(1j * 2 * np.pi / period * Xr) + \
           ampl2 * np.exp(-1j * 2 * np.pi / period * Xr) + dc_power
    pup_im = im_c * mask
    # Total pupil power, normalised such that = 1 for unity-filled aperture:
    puppower_adj = np.sum(np.abs(pup_im)**2) / np.sum(mask) # Total pupil power, normalised
    return pup_im, puppower_adj, mask


# # Set speckle parameters
# dc_power = 2
# # period=64
# cycles_per_pupil = 4
# ampl1=1.
# ampl2=0.5
# rot=30
# phi = 0
#
# pup_im, puppower_adj, mask = make_virt_pupil(dc_power, cycles_per_pupil, ampl1, ampl2, rot, phi)


# Set speckle parameters
dc_power = [1.5, 0]
cycles_per_pupil = [4, 6]
ampl1 = [1., 0.7]
ampl2 = [0.5, 0.3]
rot = [0, 45]
# phi = [0, 30]
phi = [0, 15]
nspeck = len(dc_power)

pup_im = np.zeros((npx_pup, npx_pup), dtype='complex')
puppower_adj = 0
for k in range(nspeck):
    pup_im_cur, puppower_adj_cur, mask = make_virt_pupil(dc_power[k], cycles_per_pupil[k], ampl1[k], ampl2[k],
                                                 rot[k], phi[k])
    pup_im = pup_im + pup_im_cur
    puppower_adj = puppower_adj + puppower_adj_cur




# Make PSF from virtual pupil
pup_padded = np.pad(pup_im, padding_px)
psf_raw = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup_padded), norm='backward'))
psf_raw = psf_raw / np.sqrt(psf_raw.shape[0]*psf_raw.shape[1]) #Normalise FT by area
psf_intens = np.abs(psf_raw)**2
impower_adj = np.sum(psf_intens) / np.sum(mask)

if cropsize is not None:
    cnt = psf_intens.shape[0] // 2
    hw = cropsize // 2
    psf_intens = psf_intens[cnt - hw:cnt + hw + 1, cnt - hw:cnt + hw + 1]
    psf_raw = psf_raw[cnt - hw:cnt + hw + 1, cnt - hw:cnt + hw + 1]


plt.figure(1)
plt.clf()
plt.subplot(221)
plt.imshow(np.abs(pup_im))
plt.colorbar()
plt.title('Pupil amplitude')
plt.text(0.05, 0.05, 'Total power (adj): %.3f' % puppower_adj,
         transform=plt.gca().transAxes, color='white')
plt.subplot(222)
plt.imshow(np.angle(pup_im), cmap='hsv')
plt.colorbar()
plt.title('Pupil phase')

plt.subplot(223)
vmax = np.max(psf_intens) * 0.5
# vmax = None
plt.imshow(psf_intens, cmap='turbo', vmax=vmax)
# plt.imshow(np.log10(psf_intens), vmin=0)
# plt.imshow(np.sqrt(psf_intens))
plt.colorbar()
plt.title('PSF intensity (virt pupil)')
plt.text(0.05, 0.05, 'Total power (adj): %.3f' % impower_adj,
         transform=plt.gca().transAxes, color='white')
plt.subplot(224)
plt.imshow(np.angle(psf_raw), cmap='hsv')
plt.colorbar()
plt.title('PSF phase (virt pupil)')
# plt.tight_layout()



# Try adding diffraction grid
grid_cellsz = 1
grid_padding = 256 # Before scaling
# grid_rescale = 1

pupim_forgrid = np.pad(pup_im, grid_padding)
mask_forgrid = np.pad(mask.astype('float'), grid_padding)
# pupim_forgrid_rsz_rl = rescale(np.real(pupim_forgrid), grid_rescale, order=1)
# pupim_forgrid_rsz_im = rescale(np.imag(pupim_forgrid), grid_rescale, order=1)
# pupim_forgrid_rsz = pupim_forgrid_rsz_rl + 1j*pupim_forgrid_rsz_im
pupim_forgrid_rsz = pupim_forgrid
pupgrid_imsz = pupim_forgrid_rsz.shape[0]
if pupgrid_imsz % (grid_cellsz * 2) !=0:
    print('Error - pupil size not a multiple of grid cell size')
ncells = int(pupgrid_imsz / (grid_cellsz * 2))
checkerboard = np.kron([[1, 0] * ncells, [0, 1] * ncells] * ncells, np.ones((grid_cellsz, grid_cellsz))) * np.pi

amp_inv = -(np.abs(pupim_forgrid_rsz) / np.max(np.abs(pupim_forgrid_rsz))) + 1
# amp_inv = -np.abs(pupim_forgrid_rsz)
slmim = np.angle(pupim_forgrid_rsz) + amp_inv*checkerboard
slmim_ph = mask_forgrid * np.exp(1j * slmim)

# Simple make psf
psfim_c = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(slmim_ph), norm='backward'))
psfim_c = psfim_c / np.sqrt(psfim_c.shape[0]*psfim_c.shape[1]) #Normalise FT by area
psfim_intens = np.abs(psfim_c)**2

if simpsf_cropsize is not None:
    cnt = psfim_intens.shape[0] // 2
    hw = simpsf_cropsize // 2
    psfim_intens = psfim_intens[cnt - hw:cnt + hw + 1, cnt - hw:cnt + hw + 1]


plt.figure(2)
plt.clf()
plt.subplot(121)
vmax = np.max(psfim_intens) * 0.5
plt.imshow(psfim_intens, cmap='turbo', vmax=vmax)
plt.subplot(122)
plt.imshow(slmim)







