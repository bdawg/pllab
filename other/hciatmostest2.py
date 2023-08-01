from hcipy import *
from scipy.ndimage.measurements import center_of_mass as com
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pickle

def velocity(speed:'ms', direction:'rad'):
    return (speed*np.cos(direction), speed*np.sin(direction))

D_tel = 8 # meter
wavelength = 1e-6 # meter

pupil_grid = make_pupil_grid(512, D_tel)
focal_grid = make_focal_grid_from_pupil_grid(pupil_grid, 8, 16*10, wavelength=wavelength)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

aperture = circular_aperture(D_tel)(pupil_grid)

fried_parameter = 0.2 # meter
outer_scale = 2 # meter

Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
wf = Wavefront(aperture, wavelength)
# the velocity is (speed, angle (rads))
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity = velocity(6.5, np.pi/4))

imshow_field(layer.phase_for(wavelength), cmap='RdBu', vmin = -5, vmax = 5)

img = prop(layer(wf))
imshow_field(np.log10(img.intensity / img.intensity.max()), vmin=-3, vmax = 3)

