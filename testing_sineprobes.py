import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, windows
plt.ion()

def make_1dgaussian(x, A, fwhm, imX):
    c = fwhm / 2.35482
    im_gauss = A * np.exp(-((imX - x) ** 2 / (2 * c ** 2)))
    return im_gauss

def make_2dgaussian(y, x, A, fwhm, imY, imX):
    c = fwhm / 2.35482
    im_gauss = A * np.exp(-((imY - y) ** 2 / (2 * c ** 2) + (imX - x) ** 2 / (2 * c ** 2)))
    return im_gauss


# ##### Image to pupil
# fwhm = 0.5
# sep1 = 50
# contr1 = 0.5
# sep2 = -50
# contr2 = 0.5
# pup_xlims = (-100,100)
# im_xlims = (-200,200)
#
# xinds = np.arange(-1000, 1000)
# im = np.zeros(xinds.shape[0])
# central_psf = make_1dgaussian(xinds, 1, fwhm, 0)
# im += central_psf
# secpsf1 = make_1dgaussian(xinds, contr1, fwhm, sep1)
# im += secpsf1
# secpsf2 = make_1dgaussian(xinds, contr2, fwhm, sep2)
# im += secpsf2
# im = im.astype('complex')
# im[1025:] = im[1025:] * np.exp(1j * 1*np.pi)
# # im[990:1010] = im[990:1010] * np.exp(1j * 0*np.pi)
# pup = np.fft.fft(np.fft.fftshift(im))

##### Pupil to image
sineAmp = 1
sinePer = 100 / (2*np.pi)
sinePhi = 0
phasestripeAmp = np.pi
im_xlims = (-200,200)
pup_xlims = None#200

xinds = np.arange(-1000, 1000)
win = windows.get_window('hamming', xinds.shape[0])
# pup = np.zeros(xinds.shape[0], dtype='complex')
patt1_phase = sineAmp * np.sin(xinds/sinePer + sinePhi)
# patt1_phase = patt1_phase**2
# xampl_rad = xinds * (1/sinePer)
# patt1_ampl = phasestripeAmp*square(xinds + sinePhi)
# patt1_ampl = phasestripeAmp*square(xinds*1/(2*np.pi))
# patt1_ampl -= (np.min(patt1_ampl)+0.1)
# patt1_ampl /= np.max(patt1_ampl)
patt1_ampl = np.ones(xinds.shape[0])
# patt1_ampl = patt1_phase**2/2
# mean_amp = 1
# pv_amp = 0.5
# ampramp1 = np.linspace(mean_amp-pv_amp/2,mean_amp+pv_amp/2,xinds.shape[0])
# ampramp1 = ampramp1 % 2
# patt1_ampl *= ampramp1

# patt1_ampl = sineAmp * np.cos(xinds/sinePer + sinePhi) + sineAmp
cosPhi=0
# for cosPhi in np.linspace(0, 2*np.pi,100):
cosPer = sinePer*1
cosAmp = sineAmp*1
patt1_ampl = cosAmp * np.sin(xinds/cosPer + cosPhi +np.pi/2) + cosAmp

# patt1_ampl *= win
patt1 = patt1_ampl * np.exp(1j * patt1_phase)

patt1 = patt1_ampl + 1j*patt1_phase

pup = patt1

im = np.fft.fft(np.fft.fftshift(pup))
im = np.fft.fftshift(im)


plt.figure(1)
plt.clf()
plt.subplot(211)
plt.plot(xinds, np.abs(im))
plt.xlim(im_xlims)
plt.title('Image amplitude')
plt.subplot(212)
plt.title('Image phase')
plt.plot(xinds, np.angle(im))
plt.xlim(im_xlims)
plt.tight_layout()
plt.pause(0.0001)

plt.figure(2)
plt.clf()
plt.subplot(211)
plt.plot(xinds, np.abs(pup), '-')
plt.title('Pupil amplitude')
plt.xlim(pup_xlims)
plt.subplot(212)
plt.title('Pupil phase')
plt.plot(xinds, np.angle(pup), '-')
plt.xlim(pup_xlims)
plt.tight_layout()

print(cosPhi)
plt.pause(0.1)
