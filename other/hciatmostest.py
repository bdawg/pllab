
from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import time




D_tel = 8.2 # meter
wavelength = 1e-6 # meter
px = 64


pupil_grid = make_pupil_grid(px, D_tel)
# focal_grid = make_focal_grid(8, 16, wavelength=wavelength)
# focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, 16, 16, wavelength=wavelength)
focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, 4, 8, wavelength=wavelength)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

aperture = circular_aperture(D_tel)(pupil_grid)

fried_parameter = 0.2 # meter
outer_scale = 20 # meter
velocity = 5 # meter/sec

Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)

# imshow_field(layer.phase_for(wavelength), cmap='RdBu')
# plt.colorbar()
# plt.show()

wf = Wavefront(aperture, wavelength)


tvals = np.arange(0,10,0.001)
nits = len(tvals)
all_wfs = np.zeros((px,px,nits))

starttime = time.time()

k = 0
for t in tvals:
    layer.t=t

    wf = layer.phase_for(wavelength).reshape(px,px)
    all_wfs[:,:,k] = wf

    # img = prop(layer(wf))
    #
    plt.clf()
    # plt.subplot(121)
    plt.imshow(wf)
    # plt.subplot(122)
    # plt.imshow(img.intensity.reshape(px,px))
    plt.pause(0.01)

    # imshow_field(layer.phase_for(wavelength), cmap='RdBu', vmin=-5, vmax=5)


    k = k+1

elapsed = time.time()-starttime

print('Time per iteration (s): %.4f' % (elapsed/nits))