import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.ion()

"""
Adapted from scexPLdata class.
Todo: merge back
"""

class PLdata:
    def __init__(self, datadir, datafilename=None, coreposns_file=None, remove_clock=True):
        self.imcube_pl = None
        self.slmims_filename = None
        self.avim = None
        self.noise_sig = None
        self.sd_im = None
        self.core_posns = None
        self.popt = None
        self.datadir = datadir

        if coreposns_file is not None:
            self.load_coreposns(coreposns_file)
            self.coreposns_file = coreposns_file

        if datafilename is not None:
            self.load_pldata_file(datafilename=datafilename, remove_clock=remove_clock)


    def load_pldata_file(self, datafilename, remove_clock=True, frame_lim=5000):
        print('Loading ' + datafilename)
        npf = np.load(self.datadir + datafilename, allow_pickle=True)
        imcube_pl = npf['imcube_cam1']
        darkframes = npf['darkframes']
        all_slmim_params = npf['all_slmim_params'].item()
        if frame_lim is not None:
            imcube_pl = imcube_pl[:frame_lim,:]
        self.imcube_pl = imcube_pl - darkframes[1]
        if remove_clock:
            self.imcube_pl[:, 0, :] = 0
        self.slmims_filename = npf['slmims_filename'].item()
        self.avim = np.mean(self.imcube_pl, 0)
        self.sd_im = np.std(self.imcube_pl, 0)
        self.noise_sig = np.median(self.sd_im)
        self.all_slmim_params = all_slmim_params


    def find_core_locations_mono(self, p0, grid_rad_px_factor=4, fwhm=5, hex_range=(-5, 5), ampl=1,
                                 excl_inds=(0, 4, 34, 60, 56, 26), coreposns_file=None, save=False,
                                 show_p0_only=False, savedir=None):
        """
        p0: [edge_length, rotate, offset, aspect]
        grid_rad_px_factor: grid_rad_px = grid_rad_px_factor * edge_length
        """
        if self.avim is None:
            print('Error: no reference image loaded.')
            return

        if savedir is None:
            savedir = self.datadir

        def hex_center(x, y, z, edge_length):
            return ((1 * x - 0.5 * y - 0.5 * z) * edge_length,
                    (np.sqrt(3) / 2 * y - np.sqrt(3) / 2 * z) * edge_length)

        def make_hexgrid(hex_range, edge_length, grid_rad_px, rotate=0, offset=None,
                         aspect=None, excl_inds=None):
            hex_y, hex_x = np.mgrid[hex_range[0]:hex_range[1], hex_range[0]:hex_range[1]]
            y, x = hex_center(hex_y, hex_x, 0, edge_length)
            x = x.ravel()
            y = y.ravel()
            r = np.sqrt(x ** 2 + y ** 2)
            ok = r <= grid_rad_px
            x = x[ok]
            y = y[ok]

            if excl_inds is not None:
                try:
                    x = np.delete(x, excl_inds)
                    y = np.delete(y, excl_inds)
                except:
                    pass

            if rotate != 0:
                th = rotate / 180 * np.pi
                R = np.array([[np.cos(th), -np.sin(th)],
                              [np.sin(th), np.cos(th)]])
                [y, x] = R @ [y, x]

            if aspect is not None:
                x = x * aspect

            if offset is not None:
                y = y + offset[0]
                x = x + offset[1]

            return y, x

        def make_2dgaussian(y, x, A, fwhm, imY, imX):
            c = fwhm / 2.35482
            im_gauss = A * np.exp(-((imY - y) ** 2 / (2 * c ** 2) + (imX - x) ** 2 / (2 * c ** 2)))
            return im_gauss

        def modelfunc(p, showplot=False, returnposns=False):
            edge_length = p[0]
            rotate = p[1]
            offset = [p[2], p[3]]
            aspect = p[4]
            yposn, xposn = make_hexgrid(hex_range=hex_range, edge_length=edge_length, grid_rad_px=grid_rad_px,
                                        rotate=rotate, offset=offset, aspect=aspect, excl_inds=excl_inds)

            n_elems = xposn.size
            testim = np.zeros_like(refim)
            for k in range(n_elems):
                im_gauss = make_2dgaussian(yposn[k], xposn[k], ampl, fwhm, imY, imX)
                testim += im_gauss

            fitim = (refim - testim) ** 2
            # fitim = np.abs(refim - testim)
            # fitim = (refim - testim)
            gof = np.mean(fitim)

            if showplot:
                plt.clf()
                plt.subplot(211)
                plt.imshow(refim - testim)
                plt.subplot(212)
                plt.imshow(refim)
                plt.plot(xposn, yposn, 'xw')
                for k in range(n_elems):
                    plt.text(xposn[k], yposn[k], '%d' % (k), color='white')

            if returnposns:
                return yposn, xposn
            else:
                return gof

        refim = np.sqrt(self.avim)
        refim[np.isnan(refim)] = 0
        refim /= np.max(refim)
        imY, imX = np.mgrid[0:refim.shape[0], 0:refim.shape[1]]
        grid_rad_px = grid_rad_px_factor * p0[0]

        if not show_p0_only:
            res = minimize(modelfunc, p0, options={'disp': True})
            self.popt = res.x
            yposn_opt, xposn_opt = modelfunc(self.popt, showplot=True, returnposns=True)
            self.core_posns = np.array([yposn_opt, xposn_opt])
            self.optimize_result = res

            if save:
                if coreposns_file is None:
                    print('Error: cannot save, coreposns_file not specified.')
                    return
                np.savez(savedir + coreposns_file, p_opt=self.popt, yposn=yposn_opt,
                         xposn=xposn_opt, refim=refim)
                self.coreposns_file = coreposns_file
                print('Saved fitted coordinates to ' + coreposns_file)
        else:
            yposn_opt, xposn_opt = modelfunc(p0, showplot=True, returnposns=True)


    def load_coreposns(self, coreposns_file, loaddir=None):
        if loaddir is None:
            loaddir = self.datadir
        npf = np.load(loaddir + coreposns_file)
        yposn = npf['yposn']
        xposn = npf['xposn']
        self.core_posns = np.array([yposn, xposn])
        self.coreposns_file = coreposns_file
        print('Loaded ' + coreposns_file)


    def extract_fluxes_mono(self, winsz=7, nfrms=None, showplot=False):
        nwgs = self.core_posns.shape[1]
        if nfrms is None:
            nfrms = self.imcube_pl.shape[0]
        sz = winsz//2
        self.all_subims = np.zeros((nfrms, nwgs, winsz, winsz))
        self.all_fluxes = np.zeros((nfrms, nwgs))
        for k in range(nfrms):
            if k % 1000 == 0:
                print('Extracting frame %d' % k)
            im = self.imcube_pl[k, :, :]
            for wg in range(nwgs):
                ps = self.core_posns[:,wg]
                ps = np.round(ps).astype('int')
                self.all_subims[k, wg, :, :] = im[ps[0]-sz:ps[0]+sz+1, ps[1]-sz:ps[1]+sz+1]
                self.all_fluxes[k, wg] = np.sum(self.all_subims[k, wg, :, :])
                if showplot:
                    plt.clf()
                    plt.imshow(self.all_subims[k, wg, :, :])
                    plt.title(wg)
                    plt.pause(0.05)


    def extract_fluxes_poly_simple(self, spec_length=24, spec_width=3, callaser_posn=10,
                                   nfrms=None, show_main_plot=False, show_indiv_plots=False):
        nwgs = self.core_posns.shape[1]
        if nfrms is None:
            nfrms = self.imcube_pl.shape[0]

        if show_main_plot:
            refim = np.mean(self.imcube_pl, 0)
            refim /= np.max(refim)
            xposn = self.core_posns[1, :]
            yposn = self.core_posns[0, :]
            plt.figure(1)
            plt.clf()
            plt.imshow(refim)
            plt.plot(xposn, yposn, 'xw')
            for k in range(nwgs):
                plt.text(xposn[k], yposn[k], '%d' % (k), color='white')
                plt.gca().add_patch(Rectangle((xposn[k] - callaser_posn, yposn[k] - (spec_width / 2)),
                                              spec_length, spec_width, edgecolor='white', fill=False))

        self.all_subims = np.zeros((nfrms, nwgs, spec_width, spec_length))
        self.all_fluxes = np.zeros((nfrms, nwgs, spec_length))
        self.all_vprofs = np.zeros((nfrms, nwgs, spec_width))
        hw = spec_width // 2
        for k in range(nfrms):
            if k % 1000 == 0:
                print('Extracting frame %d' % k)
            im = self.imcube_pl[k, :, :]
            for wg in range(nwgs):
                ps = self.core_posns[:,wg]
                ps = np.round(ps).astype('int')
                subim = im[ps[0] - hw:ps[0] + hw + 1, ps[1] - callaser_posn:ps[1] + (spec_length - callaser_posn)]
                self.all_subims[k, wg, :, :] = subim
                self.all_fluxes[k, wg, :] = np.sum(subim, 0)
                self.all_vprofs[k, wg, :] = np.sum(subim, 1)
                if show_indiv_plots:
                    plt.figure(2)
                    plt.clf()
                    plt.subplot(211)
                    plt.imshow(subim)
                    plt.title(wg)
                    plt.subplot(212)
                    plt.plot(self.all_fluxes[k, wg, :])
                    plt.pause(0.2)


    def extract_multifile_fluxes(self, filepref, numfiles, mode='mono', savefilename=None,
                                 appendfilename=True, skipfile=None):
        nwgs = self.core_posns.shape[1]
        multifile_fluxes_list = []
        combined_slmimparams_list = []
        combined_coeffs_list = []
        fnums = np.arange(numfiles)
        if skipfile is not None:
            fnums = fnums[fnums != skipfile]
        for k in fnums:
            filename = filepref + '%.2d' % k + '.npz'
            self.load_pldata_file(filename)
            combined_slmimparams_list.append(self.all_slmim_params)
            try:
                coeffs = self.all_slmim_params['coeffs']
            except:
                coeffs = None
            combined_coeffs_list.append(coeffs)
            if mode == 'mono':
                self.extract_fluxes_mono()
                multifile_fluxes_list.append(self.all_fluxes)
            elif mode == 'poly':
                print('Only *simple* spectral extraction implemented')
                self.extract_fluxes_poly_simple()
                multifile_fluxes_list.append(self.all_fluxes)
            else:
                print('Error: unknown mode specified')
                return

        self.multifile_fluxes = np.asarray(multifile_fluxes_list)
        combined_coeffs = np.array(combined_coeffs_list)
        try:
            nmodes = combined_coeffs.shape[2]
            combined_coeffs = np.squeeze(combined_coeffs.reshape((1, -1, nmodes)))
        except:
            print('No coeffs found')
        if mode == 'mono':
            self.multifile_fluxes = np.squeeze(self.multifile_fluxes.reshape((1,-1,nwgs)))
        elif mode == 'poly':
            spec_length = self.multifile_fluxes.shape[3]
            self.multifile_fluxes = np.squeeze(self.multifile_fluxes.reshape((1,-1,nwgs,spec_length)))

        if savefilename is not None:
            if appendfilename:
                savefileprefix = filepref[:-5]
                savefilename = savefilename + '__' + savefileprefix
            np.savez(self.datadir+savefilename, all_fluxes=self.multifile_fluxes,
                     ogfilename=filepref, coreposns_file=self.coreposns_file,
                     combined_slmimparams_list=combined_slmimparams_list, combined_coeffs=combined_coeffs)
            print('Saved %d flux measurements to ' % self.multifile_fluxes.shape[0] + savefilename)



