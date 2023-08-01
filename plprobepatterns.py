import numpy as np
import time
import matplotlib
from scipy.signal import square
# import hcipy
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()



class plprobe:
    def __init__(self, slm_centre=(553,378), slm_rad=190, slmoffset=127, D_tel=8, wl=1.5e-6,
                 init_ampl=1, showplot=True):
        # if complex:
        #     self.dtype = 'complex'
        # else:
        #     self.dtype = 'float'
        # self.dtype = 'float'
        npix = slm_rad * 2
        # self.pupil_grid = hcipy.make_pupil_grid(npix, D_tel)
        Y, X = np.mgrid[-npix / 2:npix / 2, -npix / 2:npix / 2]
        R = np.sqrt(X ** 2 + Y ** 2)
        mask = np.zeros((npix, npix), dtype='bool')
        mask[R > slm_rad] = True

        self.npix = npix
        self.mask = mask
        self.slmoffset = slmoffset
        self.wl = wl
        self.showplot = showplot
        # Initialise with constant amplitude:
        self.pupampl = np.ones((self.npix, self.npix)) * init_ampl
        self.pupphase = np.zeros((self.npix, self.npix))


    def clearim(self):
        self.pupampl = np.zeros((self.npix, self.npix))
        self.pupphase = np.zeros((self.npix, self.npix))


    def plot_probeim(self, fignum=1, phaseim=None):
        plt.figure(fignum)
        plt.clf()
        plt.subplot(121)
        plt.imshow(self.pupampl)
        plt.colorbar()
        pv_amp = np.max(self.pupampl) - np.min(self.pupampl)
        plt.title('Pupil amplitude - P-V = %.2f' % pv_amp)
        plt.subplot(122)
        lbl = ''
        if phaseim is None:
            phaseim = self.pupphase
            lbl = ' RAD'
        plt.imshow(phaseim, cmap='twilight')
        plt.colorbar()
        pv_phase = np.max(phaseim) - np.min(phaseim)
        plt.title('Pupil phase - P-V = %.2f' % pv_phase + lbl)
        plt.pause(0.001)


    def makephaseramp(self, xslope=0.1, yslope=0.1, showplot=False, apply=False, return_im=False,
                      mode='phase'):
        Y, X = np.mgrid[:self.npix, :self.npix]
        im = xslope * X + yslope * Y
        # im -= im[self.slmdims[0] // 2, self.slmdims[1] // 2]
        if showplot:
            plt.clf()
            plt.imshow(im)
            plt.colorbar()
            plt.pause(0.001)
        if apply:
            if mode == 'phase':
                print('Adding as phase')
                self.pupphase += im
            elif mode == 'ampl':
                print('Adding as amplitude')
                self.pupampl += im
            else:
                print('Unknown mode specified')
                return
        if return_im:
            return im


    def makestripes(self, period=20, ampl=np.pi, phi=0, type='square', mode='phase',
                    rot=0, showplot=False, apply=False, return_im=False):
        # x = np.arange(-self.npix/2, self.npix/2)
        # x_rad = x * (1 / period) * 2 * np.pi

        Y, X = np.mgrid[-self.npix / 2:self.npix / 2, -self.npix / 2:self.npix / 2]
        Xr = np.cos(rot/180*np.pi) * X + np.sin(rot/180*np.pi) * Y
        x_rad = Xr * (1 / period) * 2 * np.pi

        if type == 'square':
            y = square(x_rad + phi) * ampl
        elif type == 'sine':
            y = (np.sin(x_rad + phi) * ampl)
        elif type == 'sinesquared':
            y = (np.sin(x_rad + phi) * ampl) ** 2
        else:
            print('Unknown type specified')
            return
        im = np.broadcast_to(y, (self.npix, self.npix))

        if showplot:
            plt.clf()
            plt.imshow(im)
            plt.colorbar()
            plt.pause(0.001)
        if apply:
            if mode == 'phase':
                print('Adding as phase')
                self.pupphase += im
            elif mode == 'ampl':
                print('Adding as amplitude')
                self.pupampl += im
            else:
                print('Unknown mode specified')
                return
        if return_im:
            return im


    def make_asymmstripes(self, period=20, ampl1=1., ampl2=1., DC=0., apply=False,
                          rot=0., return_im=False):
        # x = np.arange(-self.npix / 2, self.npix / 2)
        # y = ampl1 * np.exp(1j * 2 * np.pi / period * x) + \
        #     ampl2 * np.exp(-1j * 2 * np.pi / period * x) + DC
        # im_c = np.broadcast_to(y, (self.npix, self.npix))

        Y, X = np.mgrid[-self.npix / 2:self.npix / 2, -self.npix / 2:self.npix / 2]
        Xr = np.cos(rot/180*np.pi) * X + np.sin(rot/180*np.pi) * Y
        im_c = ampl1 * np.exp(1j * 2 * np.pi / period * Xr) + \
                ampl2 * np.exp(-1j * 2 * np.pi / period * Xr) + DC
        if apply:
            # self.pupphase += np.angle(im_c)
            # self.pupampl += np.abs(im_c)
            # print('Currently REPLACING existing pupil image')
            # self.pupphase = np.angle(im_c)
            # self.pupampl = np.abs(im_c)

            curpup_c = self.pupampl * np.exp(1j*self.pupphase)
            # newpup_c = curpup_c * im_c
            newpup_c = curpup_c + im_c
            self.pupphase = np.angle(newpup_c)
            self.pupampl = np.abs(newpup_c)

        if return_im:
            return im_c


    def sim_psf(self, showplot=False, return_im=False, padfact=6, cropsize=500, return_complex=False):
        pup_masked = np.ones((self.npix, self.npix))
        pup_masked[self.mask] = 0
        pup_ampl = self.pupampl * pup_masked
        pup_phase = self.pupphase
        pup_complx = pup_ampl * np.exp(1j * pup_phase)
        self.pup_complx = pup_complx

        # pup_padded = np.zeros((self.npix*padfact, self.npix*padfact), dtype='complex')
        # pup_padded[:self.npix, :self.npix] = pup_complx
        padpix = (self.npix*padfact - self.npix)//2
        pup_padded = np.pad(pup_complx, padpix)
        psf_raw = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup_padded)))
        psf_intens = np.abs(psf_raw)**2

        if cropsize is not None:
            cnt = psf_intens.shape[0] // 2
            hw = cropsize // 2
            psf_intens = psf_intens[cnt-hw:cnt+hw+1, cnt-hw:cnt+hw+1]
            psf_raw = psf_raw[cnt-hw:cnt+hw+1, cnt-hw:cnt+hw+1]
        if showplot:
            plt.figure(2)
            plt.clf()
            plt.imshow(psf_intens)
            plt.colorbar()

        if return_im:
            if return_complex:
                return psf_raw
            else:
                return psf_intens


if __name__ == "__main__":
    phys_slm_rad = 190
    pix_per_super = 4
    add_one = True # Total hack... increase
    virt_slm_rad = np.round(phys_slm_rad / pix_per_super).astype(int)


    prb = plprobe(slm_rad=virt_slm_rad, init_ampl=0)

    # plt.clf()
    # plt.imshow(prb_dp.SLM_ampl)
    # plt.colorbar()


    dc_intens = 1
    ampl1_intens = 1
    ampl2_intens = 0.
    req_total_intens = 0.9
    norm_fact = (dc_intens + ampl1_intens + ampl2_intens) / req_total_intens
    dc_intens /= norm_fact
    ampl1_intens /= norm_fact
    ampl2_intens /= norm_fact

    # prb.make_asymmstripes(period=20, ampl1=1/np.sqrt(2)-0.1, ampl2=0., DC=0.5-0.1, rot=0,
    #                       apply=True, return_im=False)
    prb.make_asymmstripes(period=40, ampl1=np.sqrt(ampl1_intens), ampl2=np.sqrt(ampl2_intens),
                          DC=np.sqrt(dc_intens), rot=220, apply=True, return_im=False)
    #
    # prb.make_asymmstripes(period=70, ampl1=1/np.sqrt(6), ampl2=0, DC=0,
    #                       apply=True, return_im=False, flipXY=True)

    # prb.makestripes(period=80, ampl=np.pi, phi=0, type='sinesquared', showplot=False, apply=True,
    #                 rot=0, mode='phase')
    # prb.makestripes(period=15, ampl=np.pi/3, phi=20, type='square', showplot=False, apply=True)
    # prb.makephaseramp(xslope=0.1, yslope=0.2, apply=True, mode='phase')


    # prb.pupampl = (prb.pupampl - np.min(prb.pupampl)) / (np.max(prb.pupampl) - np.min(prb.pupampl))
    # prb.pupampl = (prb.pupampl - np.min(prb.pupampl)) / 10
    # prb.pupphase = (prb.pupphase + np.pi)# / 2*np.pi

    prb.plot_probeim()
    prb.sim_psf(showplot=True, cropsize=200)

    # plt.clf()
    # plt.imshow(np.abs(prb.pup_complx))
    # plt.imshow(np.angle(prb.pup_complx))

    ##############

    # Testing double-pixel
    from SLM_encoding_program import SLM_DPixel
    import astropy.units as u


    e_diam_pixels = 190*2
    total_pixels = 1024
    full_slmdiameter = 17.4
    radian_shift = 2*np.pi # max range of SLM


    pixel_frac = e_diam_pixels / total_pixels
    e_diam_mm = full_slmdiameter*pixel_frac
    prb_dp = SLM_DPixel(x_pixels=total_pixels,
                              y_pixels=total_pixels,
                              x_dim=full_slmdiameter * u.mm,
                              y_dim=full_slmdiameter * u.mm,
                              wavelength=1.55 * u.micron,
                              e_diam = e_diam_mm * u.mm,
                              focal_length = 400 * u.mm,
                              radian_shift = radian_shift,
                              only_in_e_diam=True,
                              pix_per_super=pix_per_super,
                              add_one=add_one
                              )

    phasemap_normd = (prb.pupphase + np.pi) / (2*np.pi)
    prb_dp.CustomAmpl(prb.pupampl)
    prb_dp.CustomPhase(phasemap_normd)

    slmloc = np.array([565, 390, 190])
    slm_centre = slmloc[:2]
    slm_rad = slmloc[2]
    y_centering = -512 + slm_centre[0]
    x_centering = -512 + slm_centre[1]

    slmim_encoded_padded = prb_dp.DoublePixelConvert()
    # prb_dp.ImageShift(x_centering, y_centering, shift_super_pixel_array=False)
    slmim_encoded_padded_shifted = np.roll(slmim_encoded_padded, (y_centering, x_centering), axis = (0, 1))
    slmim_encoded = prb_dp.UnpadEncoded()
    slmim_encoded_bytes = np.round(slmim_encoded / radian_shift * 255).astype('uint8')

    # plt.figure(3)
    # plt.clf()
    # # plt.imshow(slmim_encoded_padded_shifted)
    # # plt.imshow(slmim_encoded)
    # plt.imshow(slmim_encoded_bytes)
    # plt.colorbar()
    # print(slmim_encoded[0:2,0])
    # print(slmim_encoded_bytes[0:2,0])
    print(slmim_encoded[2,0]-slmim_encoded[0,0])
    print(slmim_encoded_bytes[2,0]*1.0-slmim_encoded_bytes[0,0]*1.0)


    full_slmim = np.ones((total_pixels, total_pixels), dtype='uint8') * 127
    full_slmim[slm_centre[0] - slm_rad:slm_centre[0] + slm_rad, \
                    slm_centre[1] - slm_rad:slm_centre[1] + slm_rad] = slmim_encoded_bytes
    full_slmim += 127

    plt.figure(3)
    plt.clf()
    plt.imshow(full_slmim)
    plt.colorbar()

    # #### Test with SLM
    # if not 'slm' in locals():
    #     from plslm import plslm
    #     lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'
    #     slm = plslm(lutfile=lutfile)
    #
    # slm.slmwrite(full_slmim, showplot=False)

    # slm_flat = np.ones((1024,1024), dtype='uint8') * 127
    # slm.slmwrite(slm_flat, showplot=False)


    ############# Poppy PSF sim
    import poppy
    import astropy.units as u
    diameter = 17.4 * u.mm
    wavelength = 1.55 * u.micron
    f_len = 400 * u.mm
    entrance_diam = 5 * u.mm  # orig = 6.25mm
    transmission = np.ones((total_pixels, total_pixels))
    x_ap_shift = 0 * u.mm
    y_ap_shift = -0 * u.mm


    Final_SLM_scaled = slmim_encoded_padded / (2*np.pi + 0) * wavelength.to(u.m).value

    frsys = poppy.FresnelOpticalSystem(name='Test', pupil_diameter=1 * diameter, beam_ratio = 0.5, npix = total_pixels)
    frwf = poppy.FresnelWavefront(beam_radius = diameter/2, oversample = 2, wavelength = wavelength, npix = total_pixels)

    """
    This section creates all of the optics that will be used in the system
    """
    lens1 = poppy.QuadraticLens(f_lens = f_len, name = "Lens 1")
    apature1 = poppy.CircularAperture(radius = (entrance_diam.to(u.m).value / 2), pad_factor = 1.5,
      name = "Lens Apature 1")

    SLM = poppy.ArrayOpticalElement(transmission=transmission, opd=Final_SLM_scaled, name='SLM transformation',
      pixelscale=(17.40 / total_pixels) * u.mm/u.pixel)#, opd = wavelength.to(u.m)/(2 * np.pi))
    tilt = poppy.TiltOpticalPathDifference(name = 'SLM_tilt', tilt_angle= 1 * u.degree, rotation=90)
    #tiltedSLM = poppy.TipTiltStage(SLM, radius = 17.4 * u.mm)
    #tiltedSLM.set_tip_tilt(tip = 10 * u.arcsec, tilt = 0 * u.arcsec)

    apature1.shift_x = x_ap_shift.to(u.m).value
    apature1.shift_y = y_ap_shift.to(u.m).value

    frsys.add_optic(apature1)#, shift_x = x_centering/1024 * 0.0174 * u.m, shift_y = -y_centering/1024 * 0.0174 * u.m))
    #frsys.add_optic(tilt, distance = 0 * u.m)
    frsys.add_optic(SLM, distance = f_len)

    frsys.add_optic(lens1, distance = f_len)
    frsys.add_detector(pixelscale= 10 * u.micron / u.pixel, fov_pixels = total_pixels * u.pixel, distance = f_len)

    final, inter = frsys.propagate(frwf, return_intermediates=True)

    plt.figure(3)
    # final.display(what = 'both', colorbar = True, vmin = 0.1, vmax=5, scale = 'log')
    final.display(what = 'intensity', colorbar = True, scale='linear')
