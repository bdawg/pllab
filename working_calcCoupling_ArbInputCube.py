from lanternfiber import lanternfiber
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale

# For 19 mode:
n_core = 1.44
n_cladding = 1.4345
wavelength = 1.5 # microns
core_radius = 32.8/2 # microns

# Scale parameters
max_r = 2 # Maximum radius to calculate mode field, where r=1 is the core diameter
npix = 200 # Half-width of mode field calculation in pixels
show_plots = True

# Input fields
inp_pix_scale = 4 # input pixels / fiber-field pixels
datapath = '/Users/bnorris/Dropbox/data/fiber-imaging/sim_data/sim_psfs/'
input_filename = 'sim_seeing-psf_roughHbandSCExAO.npy'

plot_modefields = True
save_image_sequences = False
imseq_out_dir = datapath+'imseq/'



### Make the fiber and modes
f = lanternfiber(n_core, n_cladding, core_radius, wavelength)
f.find_fiber_modes()
f.make_fiber_modes(npix=npix, show_plots=False, max_r=max_r)
modes_to_measure = np.arange(f.nmodes)

# Plot all mode fields
if plot_modefields:
    plt.figure(3)
    plt.clf()
    nplots = len(f.allmodefields_rsoftorder) #7
    zlim = 0.03
    for k in range(nplots):
        plt.subplot(5,4,k+1)
        sz = f.max_r * f.core_radius
        plt.imshow(f.allmodefields_rsoftorder[k], extent=(-sz, sz, -sz, sz), cmap='bwr',
                   vmin=-zlim, vmax=zlim)
        plt.xlabel('Position ($\mu$m)')
        plt.ylabel('Position ($\mu$m)')
        core_circle = plt.Circle((0,0), f.core_radius, color='k', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(core_circle)
        plt.title(k)
    # plt.tight_layout()

# Get input fields
input_cube = np.load(datapath+input_filename)
n_flds = input_cube.shape[0]
n_flds = 1 ## For testing - just show first one


for k in range(n_flds):
    orig_field = input_cube[k,:,:]
    resized_field_real = rescale(orig_field.real, inp_pix_scale)
    resized_field_imag = rescale(orig_field.imag, inp_pix_scale)
    resized_field = resized_field_real + resized_field_imag*1j

    input_field = resized_field
    cnt = input_field.shape[1]//2
    input_field = input_field[cnt-f.npix:cnt+f.npix, cnt-f.npix:cnt+f.npix]

    f.input_field = input_field
    f.plot_injection_field(f.input_field, show_colorbar=False, logI=True, vmin=-3)
    plt.pause(0.001)

    if save_image_sequences:
        fname = imseq_out_dir + 'injplot_%.3d' % k + '.png'
        plt.savefig(fname, bbox_inches='tight', dpi=200)

    coupling, mode_coupling, mode_coupling_complex = f.calc_injection_multi(mode_field_numbers=modes_to_measure,
                                                     verbose=True, show_plots=True, fignum=2, complex=True, ylim=0.3)
    ### The complex LP mode coefficients are in mode_coupling_complex.

    if save_image_sequences:
        fname = imseq_out_dir + 'modeplot_%.3d' % k + '.png'
        plt.savefig(fname, bbox_inches='tight', dpi=200)

    plt.pause(0.5)
