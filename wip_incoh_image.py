import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm


testmode = True
# testmode = False

lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_55deg_4pi_20231023.lut'

slm_centre = [553,378]
slm_rad = 190
slmloc=np.array([slm_centre[0], slm_centre[1], slm_rad])

# slmloc = None
slm = plslm(lutfile=lutfile, testmode=testmode, slmloc=slmloc)

plt.figure(1)

angle = -90#-68
slope_rmsrad = 2
chkbd_coeff = 0

# im_rad=slm.makeramp_rad(slope_rmsrad, angle, showplot=True, showslmplot=True, return_im=True, sendtoslm=False)
#
# checkerboard_coeff = 0.93
# checkerboard_cellsz = 8
# # ncells = int(slm_rad*2 / (checkerboard_cellsz * 2))
# ncells = int(1024 / (checkerboard_cellsz * 2))
# # checkerboard = np.kron([[1, 0] * ncells, [0, 1] * ncells] * ncells,
# #                        np.ones((checkerboard_cellsz, checkerboard_cellsz))) * np.pi - np.pi/2
# checkerboard = np.kron([[1, 0] * ncells, [0, 1] * ncells] * ncells,
#                        np.ones((checkerboard_cellsz, checkerboard_cellsz))) * np.pi
# chkbd_cur = checkerboard * (1-checkerboard_coeff)
# im_slm = slm.rad2im(chkbd_cur)
#
# # slm.nextim = im_rad
# # slm.slmwrite()
#
# plt.figure(2)
# plt.clf()
# # plt.imshow(slm.nextim)
# plt.imshow(im_slm)
# plt.colorbar()


seps = np.linspace(-3, 3, 7)
angles = np.linspace(-180, 180, 6)
angles = angles[:-1]
nloops = 1

seps = [0]
angles = [0]
intensities = np.array([1, 0.75, 0.5, 0.25, 0])
intensities = np.linspace(1,0,255)
# intensities = [1]

n_slmims = len(seps) * len(angles) * len(intensities)
slmims = np.zeros((n_slmims, pllab.slm.slmdims[0], pllab.slm.slmdims[1]))
count = 0
for angle in angles:
    for sep in seps:
        for intensity in intensities:
            slope = sep * ld2r
            # im_rad = pllab.slm.makeramp_rad(slope, angle, return_im=True)
            # im_rad = im_rad + pllab.slm.make_intensity_checkerboard(intensity)
            im_rad = pllab.slm.make_intensity_checkerboard(intensity)
            im = pllab.slm.rad2im(im_rad)

            slmims[count, :,:] = im
            count += 1
pllab.all_slmims = slmims

trace1 = slmims[:,0,0]
trace2 = slmims[:,0,8]
plt.clf()
plt.plot(trace1)
plt.plot(trace2)
