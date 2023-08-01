import numpy as np
import matplotlib.pyplot as plt


# Signal parameters
N = 1000  # number of samples
f0 = 20  # main frequency component
A1 = 1  # amplitude of positive frequency component
A2 = 1#/np.sqrt(2)  # amplitude of negative frequency component (set to half power of A1)
# A2 = 0 # Could set to zero to make left-hand peak disappear.
DC = 1#*np.sqrt(2)  # DC component (set to double power of A1)

# Fourier domain signal
t = np.linspace(0, 1, N)
y = A1 * np.exp(1j*2*np.pi*f0*t) + A2 * np.exp(-1j*2*np.pi*f0*t) + DC

# Compute FFT and power spectrum
Y = np.fft.fftshift(np.fft.fft(y))
frequencies = np.fft.fftshift(np.fft.fftfreq(N))
P = np.abs(Y)**2

# Plot stuff:
pup_xlims = (0.25,0.75)
plt.figure(1)
plt.clf()
plt.subplot(221)
plt.plot(t, np.abs(y), '-')
plt.title('Pupil amplitude')
plt.xlim(pup_xlims)
plt.subplot(223)
plt.title('Pupil phase')
plt.plot(t, np.angle(y), '-')
plt.xlim(pup_xlims)
plt.subplot(222)
plt.plot(t, np.real(y), '-g')
plt.title('Pupil Real')
plt.xlim(pup_xlims)
plt.subplot(224)
plt.title('Pupil imaginary')
plt.plot(t, np.imag(y), '-g')
plt.xlim(pup_xlims)
plt.tight_layout()

im_xlims = (-0.1, 0.1)
plt.figure(2)
plt.clf()
plt.subplot(211)
plt.plot(frequencies, P)
plt.xlim(im_xlims)
plt.title('Image amplitude')
plt.subplot(212)
plt.title('Image phase')
plt.plot(frequencies, np.angle(Y))
plt.xlim(im_xlims)
plt.tight_layout()