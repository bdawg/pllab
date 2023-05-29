import cv2 as cv
from skimage.transform import rescale
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from matplotlib.patches import Circle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()


datapath = './'
infilename = 'slmscanmap_20230519a.npz'
upsample = 1
minR = 10
maxR = 100
blur_amt = 1
clip = [0.1, 0.8]
dp = 1 # 0.5
minDist = 100
p1 = 100 # 50
p2 = 30 #20


npf = np.load(datapath+infilename)
scanmap = npf['scanmap']
meas_range = npf['meas_range']
num_lin_posns = scanmap.shape[0]
meas_posns = [np.linspace(meas_range[0], meas_range[1], num_lin_posns).astype(int),
              np.linspace(meas_range[2], meas_range[3], num_lin_posns).astype(int)]

# plt.clf()
# plt.imshow(scanmap, extent=[meas_posns[1][0], meas_posns[1][-1], meas_posns[0][-1], meas_posns[0][0]])

im = np.copy(scanmap)
if upsample > 1:
    im = rescale(im, upsample, order=3)
    minR = minR * upsample
    maxR = maxR * upsample

im = im - np.min(im)
im = im / np.max(im)
im = np.clip(im, clip[0], clip[1])
im = gaussian_filter(im, blur_amt)
im = im / np.max(im) * 255
im = np.uint8(np.around(im))

# plt.clf()
# plt.imshow(im, extent=[meas_posns[1][0], meas_posns[1][-1], meas_posns[0][-1], meas_posns[0][0]])
# plt.imshow(im)
# plt.imshow(scanmap)

circles = None
circles = cv.HoughCircles(im, cv.HOUGH_GRADIENT, dp, minDist, param1=p1, param2=p2,
                          minRadius=minR, maxRadius=maxR)
if circles is None:
    print('Error: no circles found.')

# for m in np.arange(circles.shape[1]):
#     [x, y] = circles[0, m, 0:2]
#     r = circles[0, m, 2]
#     plt.plot(x, y, 'x')
#     circ_patch = plt.Circle((x, y), r, fill=False, color='r', alpha=0.5)
#     plt.gca().add_patch(circ_patch)
#     plt.pause(0.001)
[x, y] = circles[0, 0, 0:2]
r = circles[0, 0, 2]

x_slm = meas_posns[0][np.round(x).astype(int)]
y_slm = meas_posns[1][np.round(y).astype(int)]
dx = np.mean(np.diff(meas_posns[0]))
r_slm = r*dx

plt.figure(1)
plt.clf()
plt.subplot(121)
plt.imshow(im)
plt.plot(x, y, 'xw')
circ_patch = plt.Circle((x, y), r, fill=False, color='w', alpha=0.7)
plt.gca().add_patch(circ_patch)
plt.subplot(122)
plt.imshow(scanmap)
plt.plot(x, y, 'x')
circ_patch = plt.Circle((x, y), r, fill=False, color='w', alpha=0.7)
plt.gca().add_patch(circ_patch)

plt.figure(2)
plt.clf()
plt.imshow(scanmap, extent=[meas_posns[1][0], meas_posns[1][-1], meas_posns[0][-1], meas_posns[0][0]])
plt.plot(x_slm, y_slm, 'xw')
circ_patch = plt.Circle((y_slm, x_slm), r_slm, fill=False, color='w', alpha=0.7)
plt.gca().add_patch(circ_patch)
print('[x, y, r] = [%d, %d, %.1f]' % (x_slm, y_slm, r_slm))

