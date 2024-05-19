import matplotlib
matplotlib.use('TkAgg')
# import hcipy
import matplotlib.pyplot as plt
import numpy as np
# from pllab import pllab
from plslm import plslm
import time
from scipy import ndimage


datadir = 'C:/Data/'

slm_centre = [553,378]
slm_rad = 190
ld2r = 1.7 # Image scale, expressed as pupil phase ramp RMS (rad) per lambda/D PSF shift
no_slm = False

enable_incohPSFposns = True
enable_seeing = False

### Define set of incoh PSF locations and intensities
all_seps = [[0, 1, -2], [0, -0.5, 2.5, 1]] # In l/D
all_angles = [[0, 30, -70], [0, -20, 160, -150]]
all_intensities = [[1, 0.1, 0.5], [0.5, 0.1, 0.1, 0.1]]




# num_slmims = 100
num_slmims = len(all_seps)


sendtoslm = True
sendtoslm_nloops = 1
############################################################################

def reorder_coeffs_lists(*lists):
    reordered = [
        [
            [lst[i][j] for lst in lists if j < len(lst[i])]
            for j in range(len(lists[0][i]))
        ]
        for i in range(len(lists[0]))
    ]
    return reordered
all_coeffs_incohims = reorder_coeffs_lists(all_seps, all_angles, all_intensities)

slmloc=np.array([slm_centre[0], slm_centre[1], slm_rad])
# pllab = pllab(datadir=datadir, enable_cameras=False)
# pllab.slm.slmloc = slmloc
slm = plslm(testmode=no_slm, slmloc=slmloc)

if enable_seeing:
    pass
else:
    all_seeing_slmims = None

if enable_incohPSFposns:
    all_incohPSF_slmims = []
    for k in range(num_slmims):
        cur_coeffs_incohims = np.array(all_coeffs_incohims[k])
        print(cur_coeffs_incohims)
        slm_subims = slm.make_incoh_psfs(cur_coeffs_incohims[:,0], cur_coeffs_incohims[:,1],
                                         cur_coeffs_incohims[:,2], ld2r=ld2r)
        all_incohPSF_slmims.append(slm_subims)




if (not no_slm) and sendtoslm:
    for cur_slm_subims in all_incohPSF_slmims:
        for l in range(sendtoslm_nloops):
            for k in range(cur_slm_subims.shape[0]):
                slm.slmwrite(cur_slm_subims[k, :, :])
                print(k)
                plt.pause(1)
            plt.pause(2)
        slm_flat = np.ones((1024,1024), dtype='uint8') * 127
        slm.slmwrite(slm_flat, showplot=False)



