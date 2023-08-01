import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale
import time
plt.ion()



class multi_complspeckles:
    def __init__ (self, npx_pup, pupildiam_px, padding_px, cropsize=64, randomseed=None):
        self.pupildiam_px = pupildiam_px
        self.npx_pup = npx_pup
        self.padding_px = padding_px
        self.cropsize = cropsize
        Y, X = np.mgrid[-npx_pup / 2:npx_pup / 2, -npx_pup / 2:npx_pup / 2]
        self.mgrids = (Y, X)
        R = np.sqrt(X ** 2 + Y ** 2)
        mask = np.zeros((npx_pup, npx_pup), dtype='bool')
        mask[R < pupildiam_px // 2] = True
        self.pupmask = mask
        np.random.seed(randomseed)


    def make_virt_pupil(self, dc_power, cycles_per_pupil, ampl1, ampl2, rot, phi=0):
        period = self.pupildiam_px / cycles_per_pupil
        X = np.roll(self.mgrids[1], (0, phi))
        Y = self.mgrids[0]
        Xr = np.cos(rot / 180 * np.pi) * X + np.sin(rot / 180 * np.pi) * Y
        im_c = ampl1 * np.exp(1j * 2 * np.pi / period * Xr) + \
               ampl2 * np.exp(-1j * 2 * np.pi / period * Xr) + dc_power
        pup_im = im_c * self.pupmask
        puppower_adj = np.sum(np.abs(pup_im) ** 2) / np.sum(self.pupmask)  # Total pupil power, normalised
        return pup_im, puppower_adj


    def make_virt_pupil_single(self, cycles_per_pupil, ampl, rot, phi=0):
        period = self.pupildiam_px / cycles_per_pupil
        # X = np.roll(self.mgrids[1], (0, phi))
        X = self.mgrids[1] - phi
        Y = self.mgrids[0]
        Xr = np.cos(rot / 180 * np.pi) * X + np.sin(rot / 180 * np.pi) * Y
        # im_c = ampl * np.exp(1j * 2 * np.pi / period * Xr)
        im_c = ampl**2 * np.exp(1j * 2 * np.pi / period * Xr) ##########
        pup_im = im_c * self.pupmask
        puppower_adj = np.sum(np.abs(pup_im) ** 2) / np.sum(self.pupmask)  # Total pupil power, normalised
        return pup_im, puppower_adj


    def make_speckles(self, dc_powers=None, cycles_per_pupils=None, ampl1s=None, ampl2s=None, rots=None, phis=None,
                      plot=False, output_cropped=False, normalise_pupim=False, singlespeck=False):
        pup_im = np.zeros((self.npx_pup, self.npx_pup), dtype='complex')
        puppower_adj = 0
        nspeck = len(dc_powers)
        self.indiv_pupims = [] #### Testing
        for k in range(nspeck):
            if singlespeck:
                pup_im_cur, puppower_adj_cur = self.make_virt_pupil_single(cycles_per_pupils[k],
                                                                            ampl1s[k], rots[k], phis[k])
            else:
                pup_im_cur, puppower_adj_cur = self.make_virt_pupil(dc_powers[k], cycles_per_pupils[k],
                                                                      ampl1s[k], ampl2s[k], rots[k], phis[k])
            self.indiv_pupims.append(pup_im_cur) #### Testing
            pup_im = pup_im + pup_im_cur
            # puppower_adj = puppower_adj + puppower_adj_cur
        puppower_adj = np.sum(np.abs(pup_im) ** 2) / np.sum(self.pupmask)  # Total pupil power, normalised

        if normalise_pupim:
            pup_im = pup_im / np.max(np.abs(pup_im))

        # Make PSF from virtual pupil
        pup_padded = np.pad(pup_im, self.padding_px)
        # psf_raw = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup_padded), norm='backward'))
        psf_raw = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup_padded)))
        psf_raw = psf_raw / np.sqrt(psf_raw.shape[0] * psf_raw.shape[1])  # Normalise FT by area
        psf_intens = np.abs(psf_raw) ** 2
        impower_adj = np.sum(psf_intens) / np.sum(self.pupmask)

        if output_cropped:
            cnt = psf_intens.shape[0] // 2
            hw = self.cropsize // 2
            psf_intens = psf_intens[cnt - hw:cnt + hw + 1, cnt - hw:cnt + hw + 1]
            psf_raw = psf_raw[cnt - hw:cnt + hw + 1, cnt - hw:cnt + hw + 1]

        self.pup_im = pup_im
        self.puppower_adj = puppower_adj
        self.impower_adj = impower_adj
        self.psf_intens = psf_intens
        self.psf_raw = psf_raw

        if plot:
            self.makeplots()


    def make_phaseonly(self, grid_cellsz=2, grid_padding=256, output_cropped=False, showplots=False,
                       slmplot_crop=256):
        pupim_forgrid = np.pad(self.pup_im, grid_padding)
        mask_forgrid = np.pad(self.pupmask.astype('float'), grid_padding)
        pupgrid_imsz = pupim_forgrid.shape[0]
        if pupgrid_imsz % (grid_cellsz * 2) != 0:
            print('Error - pupil size not a multiple of grid cell size')
        ncells = int(pupgrid_imsz / (grid_cellsz * 2))
        checkerboard = np.kron([[1, 0] * ncells, [0, 1] * ncells] * ncells, np.ones((grid_cellsz, grid_cellsz))) * np.pi

        amp_inv = -(np.abs(pupim_forgrid) / np.max(np.abs(pupim_forgrid))) + 1
        # amp_inv = -np.abs(pupim_forgrid_rsz)
        slmim = np.angle(pupim_forgrid) + amp_inv * checkerboard
        slmim_ph = mask_forgrid * np.exp(1j * slmim)

        psfim_c = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(slmim_ph), norm='backward'))
        psfim_c = psfim_c / np.sqrt(psfim_c.shape[0] * psfim_c.shape[1])  # Normalise FT by area
        psfim_intens = np.abs(psfim_c) ** 2

        if output_cropped:
            cnt = psfim_intens.shape[0] // 2
            hw = self.cropsize // 2
            psfim_intens = psfim_intens[cnt - hw:cnt + hw + 1, cnt - hw:cnt + hw + 1]
            psfim_c = psfim_c[cnt - hw:cnt + hw + 1, cnt - hw:cnt + hw + 1]

        self.slm_im = slmim_ph
        self.slm_psf_raw = psfim_c
        self.slm_psf_intens = psfim_intens

        if showplots:
            slmim_toplot = np.copy(slmim_ph)
            if slmplot_crop is not None:
                cnt = slmim_toplot.shape[0] // 2
                hw = slmplot_crop // 2
                slmim_toplot = slmim_toplot[cnt - hw:cnt + hw + 1, cnt - hw:cnt + hw + 1]
            plt.figure(6)
            plt.clf()
            plt.subplot(121)
            plt.imshow(np.angle(slmim_toplot), cmap='hsv')
            plt.title('SLM phase map')
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(np.abs(spk.slm_psf_raw) ** 2, cmap='turbo')
            # plt.imshow(np.angle(spk.slm_psf_raw), cmap='turbo')
            plt.title('PSF via SLM')
            plt.colorbar()
            plt.pause(0.001)


    def makeplots(self, fignum=5):
        if self.cropsize is not None:
            cnt = self.psf_intens.shape[0] // 2
            hw = self.cropsize // 2
            psf_intens = self.psf_intens[cnt - hw:cnt + hw + 1, cnt - hw:cnt + hw + 1]
            psf_raw = self.psf_raw[cnt - hw:cnt + hw + 1, cnt - hw:cnt + hw + 1]
        else:
            psf_intens = self.psf_intens
            psf_raw = self.psf_raw

        plt.figure(fignum)
        plt.clf()
        plt.subplot(221)
        plt.imshow(np.abs(self.pup_im))
        plt.colorbar()
        plt.title('Pupil amplitude')
        plt.text(0.05, 0.05, 'Total power (adj): %.3f' % self.puppower_adj,
                 transform=plt.gca().transAxes, color='white')
        plt.subplot(222)
        plt.imshow(np.angle(self.pup_im), cmap='hsv')
        plt.colorbar()
        plt.title('Pupil phase')

        plt.subplot(223)
        vmax = np.max(psf_intens) * 0.5
        vmax = None
        plt.imshow(psf_intens, cmap='turbo', vmax=vmax)
        # plt.imshow(np.log10(psf_intens), vmin=0)
        # plt.imshow(np.sqrt(psf_intens))
        plt.colorbar()
        plt.title('PSF intensity (virt pupil)')
        plt.text(0.05, 0.05, 'Total power (adj): %.3f' % self.impower_adj,
                 transform=plt.gca().transAxes, color='white')
        plt.subplot(224)
        plt.imshow(np.angle(psf_raw), cmap='hsv')
        plt.colorbar()
        plt.title('PSF phase (virt pupil)')
        plt.pause(0.1)
        # plt.tight_layout()


    def setup_fibermodes(self, n_core=1.44, n_cladding=1.4345, wavelength=1.5, core_radius=32.8/2,
                         plot_modefields=False, fignum=3, mfhx_px=200):
        # Scale parameters
        max_r = 2  # Maximum radius to calculate mode field, where r=1 is the core diameter
        npix = mfhx_px  # Half-width of mode field calculation in pixels

        from lanternfiber import lanternfiber
        ### Make the fiber and modes
        self.f = lanternfiber(n_core, n_cladding, core_radius, wavelength)
        self.f.find_fiber_modes()
        self.f.make_fiber_modes(npix=npix, show_plots=False, max_r=max_r)
        self.modes_to_measure = np.arange(self.f.nmodes)

        if plot_modefields:
            plt.figure(fignum)
            plt.clf()
            nplots = len(self.f.allmodefields_rsoftorder)  # 7
            zlim = 0.03
            for k in range(nplots):
                # plt.subplot(5,4,k+1)
                plt.subplot(5, 6, k + 1)
                sz = self.f.max_r * self.f.core_radius
                plt.imshow(self.f.allmodefields_rsoftorder[k], extent=(-sz, sz, -sz, sz), cmap='bwr',
                           vmin=-zlim, vmax=zlim)
                plt.xlabel('Position ($\mu$m)')
                plt.ylabel('Position ($\mu$m)')
                core_circle = plt.Circle((0, 0), self.f.core_radius, color='k', fill=False, linestyle='--', alpha=0.5)
                plt.gca().add_patch(core_circle)
                plt.title(k)
            # plt.tight_layout()


    def measure_fibcoupling(self, orig_field, inp_pix_scale, plot_injfield=False, fignum_inj=3,
                            plot_coupling=False, fignum_coupling=2):

        resized_field_real = rescale(orig_field.real, inp_pix_scale)
        resized_field_imag = rescale(orig_field.imag, inp_pix_scale)
        resized_field = resized_field_real + resized_field_imag * 1j
        input_field = resized_field
        cnt = input_field.shape[1] // 2
        input_field = input_field[cnt - self.f.npix:cnt + self.f.npix, cnt - self.f.npix:cnt + self.f.npix]
        self.f.input_field = input_field

        if plot_injfield:
            self.f.plot_injection_field(self.f.input_field, show_colorbar=False, logI=True, vmin=-3, fignum=fignum_inj)
            plt.pause(0.001)

        coupling, mode_coupling, mode_coupling_complex = self.f.calc_injection_multi(mode_field_numbers=self.modes_to_measure,
                                                                verbose=False, show_plots=plot_coupling,
                                                                fignum=fignum_coupling, complex=True,
                                                                modes_to_plot=self.modes_to_measure, ylim=0.3)
        coeff_vec = np.concatenate((np.real(mode_coupling_complex), np.imag(mode_coupling_complex)))

        return coeff_vec


    def make_multi_simdata(self, num_data, nspecks, dc_power_range, cycles_per_pupil_range, ampl1_range, ampl2_range,
                           rot_range, phi_range, inp_pix_scale, only_one_dc=True, showplots=False,
                           savefile=None, savedir='', normalise_pupim=False, make_slmims=False, singlespeck=False,
                           save_reduced=False):
        print('Generating %d simulated data measurements' % num_data)
        all_speck_params = []
        all_pupims = []
        all_psfs = []
        all_mode_coeffs = []
        all_slmims = []
        all_slm_psfs = []

        starttime = time.time()
        for k in range(num_data):
            if k % 50 == 0:
                print('Iteration %d' % k)
            dc_powers = np.random.uniform(low=dc_power_range[0], high=dc_power_range[1], size=nspecks)
            cycles_per_pupils = np.random.uniform(low=cycles_per_pupil_range[0], high=cycles_per_pupil_range[1], size=nspecks)
            ampl1s = np.random.uniform(low=ampl1_range[0], high=ampl1_range[1], size=nspecks)
            ampl2s = np.random.uniform(low=ampl2_range[0], high=ampl2_range[1], size=nspecks)
            rots = np.random.uniform(low=rot_range[0], high=rot_range[1], size=nspecks)
            phis = np.random.uniform(low=phi_range[0], high=phi_range[1], size=nspecks).astype('int')
            if only_one_dc:
                dc_powers[1:] = 0
            if singlespeck:
                speck_params = np.array([cycles_per_pupils, ampl1s, rots, phis])
            else:
                speck_params = np.array([dc_powers, cycles_per_pupils, ampl1s, ampl2s, rots, phis])
            all_speck_params.append(speck_params)

            self.make_speckles(dc_powers, cycles_per_pupils, ampl1s, ampl2s, rots, phis, output_cropped=True,
                              plot=showplots, normalise_pupim=normalise_pupim, singlespeck=singlespeck)
            all_pupims.append(self.pup_im.astype('complex64'))
            all_psfs.append(self.psf_raw.astype('complex64'))

            mode_coeffs = self.measure_fibcoupling(self.psf_raw, inp_pix_scale=inp_pix_scale,
                                                   plot_injfield=showplots, plot_coupling=showplots)
            if showplots:
                plt.pause(0.001)
            all_mode_coeffs.append(mode_coeffs)

            if make_slmims:
                self.make_phaseonly(grid_cellsz=2, grid_padding=256, output_cropped=True, showplots=showplots)
                all_slmims.append(self.slm_im)
                all_slm_psfs.append(self.slm_psf_raw)
            if showplots:
                plt.pause(0.1)

        self.all_speck_params = np.array(all_speck_params)
        self.all_pupims = np.array(all_pupims)
        self.all_psfs = np.array(all_psfs)
        self.all_mode_coeffs = np.array(all_mode_coeffs)
        self.all_slmims = np.array(all_slmims)
        self.all_slm_psfs = np.array(all_slm_psfs)

        print('Elapsed time: %d seconds' % (time.time()-starttime))

        if savefile is not None:
            if save_reduced:
                psf_cropsize = 32
                pupil_rescale = 0.2
                pupil_cropsize = 64
                slmim_cropsize = 270
                slmpsf_cropsize = psf_cropsize

                print('Cropping PSF')
                psfs = self.all_psfs
                cnt = psfs.shape[1] // 2
                hw = psf_cropsize // 2
                psfs = psfs[:, cnt - hw:cnt + hw, cnt - hw:cnt + hw]
                # psfs = np.hstack((np.real(psfs), np.imag(psfs)))
                all_psfs_out = psfs

                print('Scaling & cropping pupil images...')
                pupims = self.all_pupims
                ndata = pupims.shape[0]
                newsz = int(pupims.shape[1] * pupil_rescale)
                pupims_rs_rl = np.zeros((ndata, newsz, newsz), dtype='float32')
                pupims_rs_im = np.zeros((ndata, newsz, newsz), dtype='float32')
                for k in range(ndata):
                    if k % 500 == 0:
                        print('Resizing pupil im %d' % k)
                    pupims_rs_rl[k, :, :] = rescale(pupims.real[k, :, :], pupil_rescale)
                    pupims_rs_im[k, :, :] = rescale(pupims.imag[k, :, :], pupil_rescale)
                pupims_rs = pupims_rs_rl + 1j * pupims_rs_im
                cnt = pupims_rs.shape[1] // 2
                hw = pupil_cropsize // 2
                pupims_rs = pupims_rs[:, cnt - hw:cnt + hw, cnt - hw:cnt + hw]
                # pupims_rs = np.hstack((np.real(pupims_rs), np.imag(pupims_rs)))
                all_pupils_out = pupims_rs
                del pupims, pupims_rs_rl, pupims_rs_im

                if make_slmims:
                    # Cropping SLM im
                    print('Cropping SLM im')
                    all_slmims_out = np.angle(self.all_slmims).astype('float32')
                    cnt = all_slmims_out.shape[1] // 2
                    hw = slmim_cropsize // 2
                    all_slmims_out = all_slmims_out[:, cnt - hw:cnt + hw, cnt - hw:cnt + hw]

                    print('Cropping SLM PSF')
                    slmpsfs = self.all_slm_psfs
                    cnt = slmpsfs.shape[1] // 2
                    hw = slmpsf_cropsize // 2
                    slmpsfs = slmpsfs[:, cnt - hw:cnt + hw, cnt - hw:cnt + hw]
                    # psfs = np.hstack((np.real(psfs), np.imag(psfs)))
                    all_slm_psfs_out = slmpsfs
                else:
                    all_slmims_out = None
                    all_slm_psfs_out = None

            else:
                all_psfs_out = self.all_psfs
                all_pupils_out = self.all_pupims
                all_slmims_out = self.all_slmims
                all_slm_psfs_out = self.all_slm_psfs

            print('Saving to ' + savedir + savefile)
            np.savez(savedir + savefile, all_speck_params=self.all_speck_params, all_pupims=all_pupils_out,
                     all_psfs=all_psfs_out, all_mode_coeffs=self.all_mode_coeffs, all_slmims=all_slmims_out,
                     all_slm_psfs=all_slm_psfs_out)
            # np.savez_compressed(savedir + savefile, all_speck_params=self.all_speck_params, all_pupims=self.all_pupims,
            #          all_psfs=self.all_psfs, all_mode_coeffs=self.all_mode_coeffs, all_slmims=self.all_slmims)
            print('Saving done.')




npx_pup = 512
pupildiam_px = 256
padding_px = 256
cropsize = 64 # For plotting
inp_pix_scale = 8 # input pixels / fiber-field pixels
mfhx_px = 200

# npx_pup = 512 //2
# pupildiam_px = 256 //2
# padding_px = 256 //4
# cropsize = 48
# inp_pix_scale = 10  # input pixels / fiber-field pixels


# # For 19 mode:
# wavelength = 1.5 # microns
# core_radius = 20.8/2 # microns
# inp_pix_scale = 4 # input pixels / fiber-field pixels

# # ~input of oversampled PL
wavelength = 1.55 # microns
core_radius = 39.4/2 # microns



# # Set speckle parameters
# dc_powers = [1.5, 0]
# cycles_per_pupils = [4, 6]
# ampl1s = [1., 0.7]
# ampl2s = [0.5, 0.3]
# rots = [0, 45]
# phis = [0, 15]
#
# dc_powers = [1, 0, 0, 0]
# cycles_per_pupils = [1, 2, 4, 8]
# ampl1s = [1, 1, 1, 1]
# ampl2s = [0.5, 0.5, 0.5, 0.5]
# rots = [0, 20, 40, 60]
# phis = [0, 0, 0, 0]
#
# dc_powers = [1]
# cycles_per_pupils = [2.5, 2, 1]
# ampl1s = [1, 1, 1, 1]
# ampl2s = [1, 1, 1, 1]
# rots = [0, 30, 60, 90]
# phis = [0, 0, 0, 0]


# spk.make_speckles(dc_powers, cycles_per_pupils, ampl1s, ampl2s, rots, phis, output_cropped=True,
#                   plot=True)
# spk.measure_fibcoupling(spk.psf_raw, inp_pix_scale=inp_pix_scale,
#                         plot_injfield=True, plot_coupling=True)

singlespeck = False
random_seed = None
spk = multi_complspeckles(npx_pup, pupildiam_px, padding_px, cropsize, randomseed=random_seed)
spk.setup_fibermodes(core_radius=core_radius, wavelength=wavelength, plot_modefields=False, mfhx_px=mfhx_px)


savedir = '../pllab_data/'
savedir = '/Users/bnorris/DontBackup/PL/'
savedir = '/media/data/bnorris/pl_simdata/'
savefile = None
# savefile = 'iminjout_testsave3.npz'

dc_power_range = [0.1,1]
cycles_per_pupil_range = [1, 2.5]
ampl1_range = [0.1, 1]
ampl2_range = [0.1, 1]
rot_range = [0, 180]
phi_range = [0,100]


# Try single-speckle mode
singlespeck = True
cycles_per_pupil_range = [0, 2.5]
# cycles_per_pupil_range = [0, 0]
ampl1_range = [0.1, 1]
rot_range = [-180, 180]
phi_range = [0,100]


num_data = 10000
nspecks = 2
showplots = False
make_slmims = False

spk.make_multi_simdata(num_data, nspecks, dc_power_range, cycles_per_pupil_range, ampl1_range, ampl2_range,
                       rot_range, phi_range, inp_pix_scale, only_one_dc=True, showplots=showplots,
                       savefile=savefile, savedir=savedir, make_slmims=make_slmims, singlespeck=singlespeck,
                       save_reduced=True)



for l in range(10):
    savefile = 'siminjout_2ssp_ampl01-1_20230728c-01_%.2d.npz' % l
    spk = multi_complspeckles(npx_pup, pupildiam_px, padding_px, cropsize, randomseed=random_seed)
    spk.setup_fibermodes(core_radius=core_radius, wavelength=wavelength, plot_modefields=False)
    spk.make_multi_simdata(num_data, nspecks, dc_power_range, cycles_per_pupil_range, ampl1_range, ampl2_range,
                           rot_range, phi_range, inp_pix_scale, only_one_dc=True, showplots=showplots,
                           savefile=savefile, savedir=savedir, make_slmims=make_slmims, singlespeck=singlespeck,
                           save_reduced=True)


# for l in range(10):
#     nspecks = 1
#     savefile = 'siminjout_1sp_ospl_20230725-01_%.2d.npz' % l
#     spk = multi_complspeckles(npx_pup, pupildiam_px, padding_px, cropsize, randomseed=random_seed)
#     spk.setup_fibermodes(core_radius=core_radius, wavelength=wavelength, plot_modefields=False)
#     spk.make_multi_simdata(num_data, nspecks, dc_power_range, cycles_per_pupil_range, ampl1_range, ampl2_range,
#                            rot_range, phi_range, inp_pix_scale, only_one_dc=True, showplots=showplots,
#                            savefile=savefile, savedir=savedir, make_slmims=make_slmims, singlespeck=singlespeck,
#                            save_reduced=True)
#
# for l in range(10):
#     nspecks = 10
#     savefile = 'siminjout_10sp_ospl_20230725-01_%.2d.npz' % l
#     spk = multi_complspeckles(npx_pup, pupildiam_px, padding_px, cropsize, randomseed=random_seed)
#     spk.setup_fibermodes(core_radius=core_radius, wavelength=wavelength, plot_modefields=False)
#     spk.make_multi_simdata(num_data, nspecks, dc_power_range, cycles_per_pupil_range, ampl1_range, ampl2_range,
#                            rot_range, phi_range, inp_pix_scale, only_one_dc=True, showplots=showplots,
#                            savefile=savefile, savedir=savedir, make_slmims=make_slmims, singlespeck=singlespeck,
#                            save_reduced=True)



