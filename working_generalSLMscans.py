
import numpy as np
from os.path import splitext
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from pllab import pllab
import time

#### Set up required parameters
# datadir = '../pllab_data/'
datadir = 'C:/Data/'

# savedatadir = 'C:/Data/'
savedatadir = datadir

darkpath = '../pllab_data/'
# lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'
# lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT'
lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_55deg_4pi_20231023.lut'

darkfile = None
darkfile = 'darks_20230628a_superK_1.npz'
darkfile = 'darks_20231012a_laser_1.npz'
take_new_darks = False

cube_nims = 10000 #10000 # Max number of frames in a cube - cannot be changed without restarting (is used to allocate shm)
camstosave = ['psf','pl'] # camera_index will be assigned in this order
cam_tints = [0.00002, 0.00005] # Laser
# cam_tints = [0.00005, 0.0005] # SuperK
cropdims = [[224, 511, 108, 395], # PSF cam, covers out to 16-pixel-period diffraction
            [192, 383, 68, 227]] # PL cam, oversized

# insert_as_subimage = False
# slmloc = None # Use SLM region location from SLM cube file
# slmloc = np.array([565, 390, 190])
slm_centre = [553,378]
slm_rad = 190
slmloc=np.array([slm_centre[0], slm_centre[1], slm_rad])
enable_cameras = True


pllab = pllab(datadir=datadir, camstosave=camstosave, lutfile=lutfile, winparams=None,
              cam_settings=None, verbose=True, cube_nims=cube_nims, shm_mode=True,
              cropdims=cropdims, darkpath=darkpath, darkfile=darkfile, delays=(48,3),
              enable_cameras=enable_cameras)
pllab.slm.slmloc = slmloc

if enable_cameras:
    for k in range(len(camstosave)):
        cmd_str = 'set tint %f' % cam_tints[k]
        pllab.send_shm_camcommand(cam_index=k, cmd_string=cmd_str)

if take_new_darks:
    ## To take darks (make sure light source is off!)
    pllab.take_darks(darkfile=darkfile, save=True)
else:
    pllab.load_darks()


#################################################################################

ld2r = 1.7 # Image scale, expressed as pupil phase ramp RMS (rad) per lambda/D PSF shifty

# Test - Make a (polar) grid of spots out to 3 l/D
seps = np.linspace(-3, 3, 7)
angles = np.linspace(-180, 180, 6)
angles = angles[:-1]
nloops = 1

seps = [0]
angles = [0]
intensities = np.array([1, 0.75, 0.5, 0.25, 0])
intensities = np.linspace(1,0,100)

# intensities = [1]
# intensities = np.arcsin(np.linspace(0, np.pi/2, 20))

n_slmims = len(seps) * len(angles) * len(intensities)
slmims = np.zeros((n_slmims, pllab.slm.slmdims[0], pllab.slm.slmdims[1]))
slmims_rad = np.zeros((n_slmims, pllab.slm.slmdims[0], pllab.slm.slmdims[1]))
int_ins = np.zeros(n_slmims)
count = 0
for angle in angles:
    for sep in seps:
        for intensity in intensities:
            slope = sep * ld2r
            # im_rad = pllab.slm.makeramp_rad(slope, angle, return_im=True)
            # im_rad = im_rad + pllab.slm.make_intensity_checkerboard(intensity)

            intensity_in = intensity
            # intensity_in = (np.arcsin(2*intensity-1) + np.pi/2) / np.pi
            int_ins[count] = intensity_in
            im_rad = pllab.slm.make_intensity_checkerboard(intensity_in)
            im = pllab.slm.rad2im(im_rad)
            slmims[count, :, :] = im
            slmims_rad[count, :,:] = im_rad
            count += 1
pllab.all_slmims = slmims

trace1 = slmims[:,0,0]
trace2 = slmims[:,0,8]
# trace1 = slmims_rad[:,0,0]
# trace2 = slmims_rad[:,0,8]
plt.clf()
# plt.plot(trace1)
# plt.plot(trace2)
plt.plot(trace1-trace2, '-+')

# pllab.show_slm_seq(waittime=0.5, current_cube_nims=n_slmims*nloops, showplot=False)
# slm_flat = np.ones((1024,1024), dtype='uint8') * 127
# pllab.slm.slmwrite(slm_flat, showplot=False)

all_imdata = pllab.run_measurements_shm(return_data=True, current_cube_nims=n_slmims*nloops)
slm_flat = np.ones((1024,1024), dtype='uint8') * 127
pllab.slm.slmwrite(slm_flat, showplot=False)

# plt.figure(1)
# for k in range(n_slmims*nloops):
#     plt.clf()
#     plt.imshow(all_imdata[0][k,:,:])
#     # pllab.show_ims(ncams=1)
#     plt.pause(1)


winparams_fluxsum_cam0 = [148, 106, 24] # Zero order 20231023 55deg
plt.figure(10)
pllab.winparams_fluxsum[0] = winparams_fluxsum_cam0
pllab.all_imcubes[0][:,:,0] = 0
all_fluxes = pllab.imfluxes(window=True, showplot=True)

fluxes_rs = all_fluxes[0].reshape(nloops,n_slmims).T
fluxes = np.mean(fluxes_rs,1)

# plt.figure(1)
# plt.clf()
# plt.plot(intensities, int_ins)

plt.figure(2)
plt.clf()
# plt.plot(fluxes_rs)
plt.plot(intensities,fluxes, '-')


# raw_ims = all_imdata[0]
# imsz = raw_ims.shape[1]

saveoutput=False
if saveoutput:
    savefile = 'intensgrid_scan0to100_raw.npz'
    np.savez('./'+ savefile+'.npz', intensities=intensities, fluxes=fluxes, fluxes_rs=fluxes_rs, lutfile=lutfile)


# tr = trace1-trace2
# f = np.zeros(64)
# for k in range(64):
#     f[k] = np.sum(tr == k)
# plt.clf()
# plt.plot(f)

