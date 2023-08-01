import matplotlib.pyplot as plt
import numpy as np
plt.ion()

npx_pup = 1024
pupildiam_px = 512
padding_px = 256
cropsize = 64 # For plotting

# Set speckle parameters
dc_power = 2
cycles_per_pupil = 4
ampl1 = 1.
ampl2 = 0.5
rot = 30
phi = 20


# Make a pupil mask
Y, X = np.mgrid[-npx_pup / 2:npx_pup / 2, -npx_pup / 2:npx_pup / 2]
R = np.sqrt(X ** 2 + Y ** 2)
mask = np.zeros((npx_pup, npx_pup), dtype='bool')
mask[R < pupildiam_px//2] = True

# # Make virtual complex pupil - simplest example
# period = pupildiam_px / cycles_per_pupil
# Y, X = np.mgrid[-npx_pup / 2:npx_pup / 2, -npx_pup / 2:npx_pup / 2]
# im_c = ampl1 * np.exp(1j * 2 * np.pi / period * X) + \
#        ampl2 * np.exp(-1j * 2 * np.pi / period * X) + dc_power
# pup_im = im_c * mask

# Make virtual complex pupil - example with rotation and phase offset
period = pupildiam_px / cycles_per_pupil
Y, X = np.mgrid[-npx_pup / 2:npx_pup / 2, -npx_pup / 2:npx_pup / 2]
X = np.roll(X, (0, phi))
Xr = np.cos(rot / 180 * np.pi) * X + np.sin(rot / 180 * np.pi) * Y
# Xr = X
im_c = ampl1 * np.exp(1j * 2 * np.pi / period * Xr) + \
       ampl2 * np.exp(-1j * 2 * np.pi / period * Xr) + dc_power
pup_im = im_c * mask

# Make PSF from virtual pupil
pup_padded = np.pad(pup_im, padding_px)
psf_raw = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup_padded)))
psf_intens = np.abs(psf_raw)**2


# Plot results
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
plt.subplot(222)
plt.imshow(np.angle(pup_im), cmap='hsv')
plt.colorbar()
plt.title('Pupil phase')

plt.subplot(223)
plt.imshow(psf_intens, cmap='turbo')
plt.colorbar()
plt.title('PSF intensity (virt pupil)')
plt.subplot(224)
plt.imshow(np.angle(psf_raw), cmap='hsv')
plt.colorbar()
plt.title('PSF phase (virt pupil)')










