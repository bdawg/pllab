import matplotlib
# matplotlib.use('TkAgg')
# import hcipy
import matplotlib.pyplot as plt
import numpy as np
import time

def rad2slm(rad_im, slmrange=2*np.pi):
    slm_im = rad_im / slmrange * 255 + 127
    slm_im = np.array(slm_im).astype('uint8')
    return slm_im

### Define active SLM region
slm_centre = [553,378]
slm_rad = 190
D_tel = 8 #meter
wavelength = 1.5e-6 # meter
slm_dim = 1024

# add_127 = True # Hack to address this calib error (or whatever) in middle SLM range (maybe due to int8 used somewhere)

save_slmcube = True
# save_slmcube = False

# savefilename = 'slmcube_20230505_seeing_0.4-10_10K_01'
# savefilename = 'slmcube_20230505_zerns_25modes_0.5-0.1_10K_01'
# savefilename = 'slmcube_202300611_zerns25m_0.5-0.1_plusseeing_0.4-10scl0.5_10K_01'
# savefilename = 'slmcube_20230505_zerns_25modes_0.4_10K_01'
savefilename = 'slmcube_20230625_complsines-01sp_04' # No '.npz'
# savefilename = 'testing3'
# datadir = '../pllab_data/'
# datadir = 'C:/Data/'
datadir = '/media/morgana2/barnaby/PL/PL_labdata/202306/slmcubes/'
# datadir = '/home/bnorris/data/'
# datadir = '/media/data/bnorris/pllab_202306/'
datadir2 = datadir

num_samps = 10000
num_files = 10


enable_zernike = False
enable_seeing = False
enable_complexsine = True

plotprobeims = None
plotprobeims = [0]
store_complexsine_psfswfs = True


### Generic
npix = slm_rad * 2
# pupil_grid = hcipy.make_pupil_grid(npix, D_tel)
Y, X = np.mgrid[-npix / 2:npix / 2, -npix / 2:npix / 2]
R = np.sqrt(X ** 2 + Y ** 2)
mask = np.zeros((npix, npix), dtype='bool')
mask[R > slm_rad] = True
rng = np.random.default_rng()

if enable_zernike:
    ### Make Zernike modes (in radians), excluding piston
    num_modes = 25
    # coefflim_1stmode = 0.5
    # coefflim_lastmode = 0.1
    coefflim_1stmode = 0.4
    coefflim_lastmode = 0.4

    zernbasis = hcipy.mode_basis.make_zernike_basis(num_modes=num_modes+1, D=D_tel, grid=pupil_grid)
    zerncube = np.zeros((num_modes, npix, npix))
    for k in range(num_modes):
        wf = np.array(zernbasis[k+1].reshape(npix,npix))
        zerncube[k,:,:] = wf

    # # View modes
    # for k in range(7):
    #     plt.clf()
    #     cur = zerncube[k,:,:]
    #     cur[mask] = np.nan
    #     plt.imshow(cur)
    #     cur_rms = np.nanstd(cur)
    #     plt.title('RMS: %.2f' % (cur_rms))
    #     plt.colorbar()
    #     plt.pause(1)
    all_coefflims = np.linspace(coefflim_1stmode, coefflim_lastmode, num_modes)
    all_coefflims[1] = all_coefflims[0] #Make tip/tilt the same
    zern_subcube = zerncube[:num_modes, :, :]
    zern_vec = zern_subcube.reshape(-1, npix ** 2)
    zern_vec = zern_vec.astype('float32')

if enable_complexsine:
    dc_power = 0.8
    num_specks = 10#18 # Each has 3 params: [period, ampl1_intens, angle]
    # paramlims_lo = [40, 0, 0]
    # paramlims_hi = [80, 0.8, 360]
    # paramlims_lo = [50, 0.1, 0]
    # paramlims_hi = [80, 0.5, 360]

    # paramlims_lo = [40, 0.1, 0]
    # paramlims_hi = [80, 1, 360]
    paramlims_lo = [60, 0, 0]
    paramlims_hi = [80, 1, 360]

    # paramlims_lo = [80, 0, 0]
    # paramlims_hi = [120, 0.7, 360]
    # paramlims_lo = [80, 0, 0, 0]
    # paramlims_hi = [120, 1, 360, 1]
    # ampl1scale = 0.3

    twosided = False
    ampl_normpk = False
    ampl_normconst = 11.15

    if twosided:
        paramlims_lo.append(paramlims_lo[1])
        paramlims_hi.append(paramlims_hi[1])
    else:
        paramlims_lo.append(0)
        paramlims_hi.append(0)
    # est_maxpupampl = (np.sqrt(dc_power) + np.sqrt(paramlims_hi[1])*num_specks)
    # print('Estimated max pupil amplitude: %.2f' % est_maxpupampl)


    print(paramlims_hi)
    print(paramlims_lo)
    phys_slm_rad = 190
    pix_per_super = 4
    add_one = True # Total hack... increase

    from plprobepatterns import plprobe
    from SLM_encoding_program import SLM_DPixel
    import astropy.units as u
    e_diam_pixels = 190*2
    total_pixels = 1024
    full_slmdiameter = 17.4
    radian_shift = 2*np.pi # max range of SLM
    pixel_frac = e_diam_pixels / total_pixels
    e_diam_mm = full_slmdiameter*pixel_frac


if enable_seeing:
    ### Make seeing
    fried_parameter = 0.4# 0.2 # meter
    outer_scale = 10 # meter
    speed = 5 # m/s
    angle = np.pi/4 # radians
    timespan = 10 # seconds
    seeing_global_scaling = 0.5 #1


    num_timesteps = num_samps
    tvals = np.linspace(0, timespan, num_timesteps)
    def velocity(speed:'ms', direction:'rad'):
        return (speed*np.cos(direction), speed*np.sin(direction))
    aperture = hcipy.circular_aperture(D_tel)(pupil_grid)
    Cn_squared = hcipy.Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
    atmlayer = hcipy.InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale,
                                           velocity = velocity(speed, angle))
    atmlayer.reset()



for fnum in range(num_files):
    all_slmim_params = {}

    if enable_zernike:
        # Zernike:
        wfcube_zern = np.zeros((num_samps, npix, npix), dtype='float32')
        coeffs = rng.uniform(low=-all_coefflims, high=all_coefflims, size=(num_samps,num_modes)).astype('float32')
        for k in range(num_samps):
            if k % 1000 == 0:
                print('Making zernike frame %d' % k)
            wf = coeffs[k,:] @ zern_vec
            wfcube_zern[k, :, :] = wf.reshape(npix,npix)

        all_slmim_params['coeffs'] = coeffs
        all_slmim_params['num_modes'] = num_modes
        all_slmim_params['coefflim_1stmode'] = coefflim_1stmode
        all_slmim_params['coefflim_lastmode'] = coefflim_lastmode
    else:
        wfcube_zern = 0

    if enable_seeing:
        # Seeing
        wfcube_seeing = np.zeros((num_samps, npix, npix), dtype='float32')
        for k in range(num_samps):
            if k % 100 == 0:
                print('Making seeing frame %d' % k)
            atmlayer.t = tvals[k]
            wf = atmlayer.phase_for(wavelength).reshape(npix, npix) * seeing_global_scaling
            wfcube_seeing[k, :, :] = wf
        all_slmim_params['r0'] = fried_parameter
        all_slmim_params['L0'] = outer_scale
        all_slmim_params['windspeed'] = speed
        all_slmim_params['windangle'] = angle
        all_slmim_params['timespan'] = timespan
        all_slmim_params['seeing_global_scaling'] = seeing_global_scaling
    else:
        wfcube_seeing = 0

    if enable_complexsine:
        everyampl = []
        wfcube_complsine = np.zeros((num_samps, npix, npix), dtype='float32')
        virt_slm_rad = np.round(phys_slm_rad / pix_per_super).astype(int)
        num_par = len(paramlims_lo)
        coeffs = np.zeros((num_samps, num_par, num_specks))
        complexsine_psfs = []
        complexsine_pupamp = []
        complexsine_pupphase = []
        for s in range(num_specks):
            cur_coeffs = rng.uniform(low=paramlims_lo, high=paramlims_hi, size=(num_samps, num_par))
            coeffs[:, :, s] = cur_coeffs

        # coeffs[0,:,0] = [60, 0.7, 30]

        dc_ampl = np.sqrt(dc_power/num_specks)
        # req_total_intens = 0.9
        # # req_total_intens /= 100
        # ampl1scale = 1
        # norm_fact = 4#1 / req_total_intens
        # dc_intens /= norm_fact

        # req_total_intens = 0.9
        # req_total_intens /= 100
        # ampl1scale = 0.03

        # req_total_intens = 0.9
        # req_total_intens /= 500
        # ampl1scale = 0.004
        for k in range(num_samps):
            if k % 100 == 0:
                print('Making complex sine frame %d' % k)
            prb = plprobe(slm_rad=virt_slm_rad, init_ampl=0)
            for s in range(num_specks):
                # Make complex pupil
                period = coeffs[k, 0, s]
                ampl1 = np.sqrt(coeffs[k, 1, s]) # * ampl1scale
                ampl2 = np.sqrt(coeffs[k, 3, s])
                angle = coeffs[k, 2, s]
                # norm_fact = (dc_intens + ampl1_intens + ampl2_intens) / req_total_intens
                # dc_intens /= norm_fact
                # ampl1_intens /= norm_fact
                # ampl2_intens /= norm_fact
                prb.make_asymmstripes(period=period, ampl1=ampl1, ampl2=ampl2,
                                      DC=dc_ampl, rot=angle, apply=True, return_im=False)
                everyampl.append([ampl1, dc_ampl])

            # Encode to SLM
            prb_dp = SLM_DPixel(x_pixels=total_pixels, y_pixels=total_pixels, x_dim=full_slmdiameter * u.mm,
                                y_dim=full_slmdiameter * u.mm, wavelength=1.55 * u.micron, e_diam=e_diam_mm * u.mm,
                                focal_length=400 * u.mm, radian_shift=radian_shift, only_in_e_diam=True,
                                pix_per_super=pix_per_super, add_one=add_one)
            phasemap_normd = (prb.pupphase + np.pi) / (2 * np.pi)
            if ampl_normconst is not None:
                prb.pupampl = prb.pupampl / ampl_normconst
            if ampl_normpk:
                prb.pupampl = prb.pupampl - np.min(prb.pupampl)
                prb.pupampl = prb.pupampl / np.max(prb.pupampl)


            prb_dp.CustomAmpl(prb.pupampl)
            prb_dp.CustomPhase(phasemap_normd)
            if plotprobeims is not None:
                if k in np.array(plotprobeims):
                    # prb.plot_probeim(fignum=2, phaseim=None)
                    prb.plot_probeim(fignum=2, phaseim=phasemap_normd)
            slmim_encoded_padded = prb_dp.DoublePixelConvert()
            slmim_encoded = prb_dp.UnpadEncoded()
            wfcube_complsine[k, :, :] = slmim_encoded
            if store_complexsine_psfswfs:
                psf = np.abs(prb.sim_psf(showplot=False, return_im=True, cropsize=50))**2
                complexsine_psfs.append(psf)
                complexsine_pupamp.append(prb.pupampl)
                complexsine_pupphase.append(prb.pupphase)


        all_slmim_params['coeffs'] = coeffs
        all_slmim_params['paramlims_lo'] = paramlims_lo
        all_slmim_params['paramlims_hi'] = paramlims_hi
        all_slmim_params['dc_ampl'] = dc_ampl
    else:
        wfcube_complsine = 0


    wfcube = wfcube_zern + wfcube_seeing + wfcube_complsine
    # Testing:
    # wfcube = np.zeros((num_samps, npix, npix), dtype='float32')

    wfcube_slm = rad2slm(wfcube)
    # if add_127:
    #     wfcube_slm += 127

    # Put into SLM cube
    # wfcube_slm = np.ones((num_samps, slm_dim, slm_dim), dtype='uint8') * 127
    # wfcube_slm[:, slm_centre[0]-slm_rad:slm_centre[0]+slm_rad, slm_centre[1]-slm_rad:slm_centre[1]+slm_rad] = \
    #     rad2slm(wfcube)
    # for k in range(10):
    #     plt.clf()
    #     cur = wfcube_slm[k,:,:]
    #     plt.imshow(cur)
    #     plt.colorbar()
    #     plt.pause(0.01)

    if save_slmcube:
        if num_files >= 1:
            cur_savefilename = savefilename + '_file%.2d' % fnum + '.npz'
        else:
            cur_savefilename = savefilename + '.npz'
        print('Saving SLM ims to ' + cur_savefilename)
        tm = time.time()
        # np.savez_compressed(datadir+cur_savefilename, all_slmims=wfcube_slm,
        #                     all_slmim_params=all_slmim_params, slmloc=np.array([slm_centre[0], slm_centre[1], slm_rad]))
        np.savez(datadir+cur_savefilename, all_slmims=wfcube_slm,
                            all_slmim_params=all_slmim_params, slmloc=np.array([slm_centre[0], slm_centre[1], slm_rad]))
        print('Time to save SLM ims (s): %.1f' % (time.time()-tm))

        if store_complexsine_psfswfs:
            cur_savefilename = savefilename + '_PSFWFs' + '_file%.2d' % fnum + '.npz'
            np.savez(datadir2 + cur_savefilename, complexsine_psfs=complexsine_psfs,
                     complexsine_pupamp=complexsine_pupamp, complexsine_pupphase=complexsine_pupphase)

plt.figure(1)
nplot = 10

for k in range(np.min((nplot, len(complexsine_psfs)))):
    plt.clf()
    # cur = wfcube_slm[k,:,:]
    cur = complexsine_psfs[k]
    plt.imshow(cur)
    plt.title('Total flux: %.2g' % np.sum(cur))
    plt.colorbar()

    # #### Test with SLM
    # slmloc = np.array([565, 390, 190])
    # slm_centre = slmloc[:2]
    # slm_rad = slmloc[2]
    # full_slmim = np.ones((total_pixels, total_pixels), dtype='uint8')
    # full_slmim[slm_centre[0] - slm_rad:slm_centre[0] + slm_rad, \
    #                 slm_centre[1] - slm_rad:slm_centre[1] + slm_rad] = wfcube_slm[k, :, :]
    # # full_slmim += 127
    # if not 'slm' in locals():
    #     from plslm import plslm
    #     lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'
    #     slm = plslm(lutfile=lutfile)
    # slm.slmwrite(full_slmim, showplot=False)

    plt.pause(0.1)
plt.figure()
plt.imshow(np.sum(complexsine_psfs,0))

complexsine_pupamp_arr = np.asarray(complexsine_pupamp)
ampl_max = np.max(complexsine_pupamp_arr,(1,2))
ampl_min = np.min(complexsine_pupamp_arr,(1,2))
plt.figure(4)
plt.clf()
plt.plot(ampl_max)
plt.plot(ampl_min)
plt.title('Max: %.3f, Min: %.3f' % (np.max(ampl_max), np.min(ampl_min)))

