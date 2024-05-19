import matplotlib.pyplot as plt
import numpy as np
from os.path import splitext
import time
import matplotlib
# matplotlib.use('TkAgg')
# from pllab import pllab

# phis = np.linspace(-4*np.pi, 4*np.pi, 1000)
# A1 = np.exp(1j*0)
# A2 = np.exp(1j*phis)
# I = np.abs(A1+A2)**2 / 4


# # I2 = 1/4 * (np.cos(phis)**2 + 2*np.cos(phis) + 1)
# phis = np.linspace(-np.pi/2, np.pi/2, 1000)
#
# I3 = (1+np.sin(phis))/2
# plt.clf()
# # plt.plot(phis,I)
# # plt.plot(phis,I2)
# plt.plot(phis,I3)



npf = np.load('intensgrid_scan0to100_raw.npz')
meas_gridintens = npf['intensities']
meas_fluxes = npf['fluxes']
meas_fluxes = meas_fluxes - np.min(meas_fluxes)
meas_fluxes = meas_fluxes / np.max(meas_fluxes)
# plt.figure(2)
# plt.clf()
# plt.plot(meas_gridintens, meas_fluxes)
#
# raw_x = np.linspace(0,1,100)
# x_rad = raw_x * np.pi - np.pi/2
# y = (1+np.sin(x_rad))/2
# plt.plot(raw_x, y)


def gridfn(x_in):
    x_rad = x_in * np.pi - np.pi / 2
    y = (1 + np.sin(x_rad)) / 2
    return y

x = np.linspace(0,1,100)
y = gridfn(x)

plt.figure(2)
plt.clf()
plt.plot(meas_gridintens, meas_fluxes)
plt.plot(x, y)

xcald = (np.arcsin(2*x-1) + np.pi/2) / np.pi
ycald = gridfn(xcald)
plt.plot(x, ycald, 'r')
