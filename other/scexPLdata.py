# Class to read raw SCExAO PL data and extract fluxes

import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
import time
plt.ion()

class scexPLdata:

    def __init__(self, datapath, datafilename, saveddarkname=None, darkfilename=None, remove_clock=True,
                 ncores=19, coreposns_file=None, timingfilename=None, numframes=None):
        self.ncores = ncores
        self.datapath = datapath
        if coreposns_file is not None:
            npfile = np.load(coreposns_file)
            self.all_posns = npfile['all_posns']
            self.all_weights = npfile['all_weights']
        if darkfilename is None:
            npfile = np.load(datapath + saveddarkname)
            self.darkframe = npfile['darkframe']
            self.noise_sig = npfile['noise_sig']
        else:
            hdulist = fits.open(datapath + darkfilename + '.fits')
            cube = hdulist[0].data
            header = hdulist[0].header
            self.darkframe = np.mean(cube, 0)

            # Now get noise map
            if remove_clock:
                cube[:, 0, :] = 0
            sd_im = np.std(cube, 0)
            self.noise_sig = np.median(sd_im)

            if saveddarkname is not None:
                print('Saving dark to ' + datapath + saveddarkname)
                np.savez(datapath + saveddarkname, darkframe=self.darkframe, header=header,
                         noise_sig=self.noise_sig)
        self.darkframe = self.darkframe.astype('float32')
        if timingfilename is None:
            timingfilename = datafilename
        print('Reading file ' + datafilename + '.fits')
        hdulist = fits.open(datapath + datafilename + '.fits')
        self.rawcube = hdulist[0].data
        self.header = hdulist[0].header
        self.timingdata = np.loadtxt(datapath + timingfilename + '.txt')
        if numframes is not None:
            self.rawcube = self.rawcube[:numframes, :, :]
            self.timingdata = self.timingdata[:numframes]
        print('Subtracting darkframe')
        self.cleancube = self.rawcube - self.darkframe
        if remove_clock:
            self.cleancube[:,0,:] = 0


    def find_core_locations(self, manual_posn_file=None, win_size=16, ampl_guess=1000, sd_guess=2):
        avim = np.mean(self.cleancube,0)
        self.avim=avim
        plt.clf()

        if manual_posn_file is None:
            plt.figure()
            plt.imshow(avim)
            print('Click on aproximate location of each core')
            all_click_posns = []
            f = plt.gcf()

            def onclick(event):
                all_click_posns.append([event.ydata, event.xdata])
                print('Position no. %d - %f, %f' % (len(all_click_posns), event.ydata, event.xdata))
                print([event.ydata, event.xdata])

            f.canvas.mpl_connect('button_press_event', onclick)

            while len(all_click_posns) < self.ncores:
                plt.pause(0.01)
            print(len(all_click_posns))

            self.all_click_posns = np.rint(all_click_posns)
            print('Done')
            plt.close()
            self.all_click_posns = np.array(self.all_click_posns, dtype=int)
            np.save('autosave_clickposns.npy', self.all_click_posns)
        else:
            self.all_click_posns = np.load(manual_posn_file)

        y, x = np.mgrid[:avim.shape[0], :avim.shape[1]]
        all_posns = []
        all_core_models = []
        all_weights = []
        all_squaremasks = []
        for k in range(self.ncores):
            mask = np.zeros_like(avim)
            c = self.all_click_posns[k,:]
            s = int(win_size//2)
            mask[c[0]-s:c[0]+s, c[1]-s:c[1]+s] = 1
            masked_im = avim#*mask
            init_model = models.Gaussian2D(amplitude=ampl_guess, x_mean=c[1], y_mean=c[0], x_stddev=sd_guess,
                                           y_stddev=sd_guess)
            fit_lm = fitting.LevMarLSQFitter()
            fitted_model = fit_lm(init_model, x, y, masked_im)
            all_posns.append([fitted_model.y_mean.value, fitted_model.x_mean.value])
            all_core_models.append(fitted_model)

            plt.clf()
            plt.imshow(masked_im)
            plt.show()
            plt.contour((fitted_model(x, y)), cmap='hot', alpha=0.5)
            plt.ylim(c[0] - s, c[0] + s)
            plt.xlim(c[1] - s, c[1] + s)
            plt.pause(0.1)

            # Now get weights for optimal extraction (see http://www.starlink.ac.uk/docs/sun45.htx/sun45se13.html)
            pE = fitted_model(x, y)
            # Mask out far-away pixels (there really aren't wings this big).
            c = np.array([fitted_model.y_mean.value, fitted_model.x_mean.value], dtype=int)
            mask = np.zeros_like(avim)
            s = int(win_size // 2)
            mask[c[0] - s:c[0] + s, c[1] - s:c[1] + s] = 1
            pE = pE * mask
            pE = pE / np.sum(pE)
            weights = pE / np.sum(pE ** 2)
            all_weights.append(weights)
            all_squaremasks.append(mask)

        all_weights = np.array(all_weights)
        all_squaremasks = np.array(all_squaremasks)
        self.all_posns = np.array(all_posns)
        self.all_core_models = all_core_models
        self.all_weights = all_weights
        self.all_squaremasks = all_squaremasks

        # np.save('autosave_trueposns.npy', all_posns)
        np.savez('autosave_coreposns.npz', all_posns=all_posns, all_weights=all_weights)

        plt.clf()
        plt.imshow(avim)
        plt.plot(self.all_posns[:, 1], self.all_posns[:, 0], 'rx')


    def quickcheck_mask(self, nfrms=1000, nits=6):
        avim = np.mean(self.cleancube[:nfrms, :, :], 0)
        weights_mask = np.sum(self.all_weights, 0)
        plt.figure()
        for k in range(nits):
            plt.clf()
            plt.imshow(avim)
            plt.pause(0.5)
            plt.clf()
            plt.imshow(avim * weights_mask)
            plt.pause(0.5)


    def extract_fluxes(self, savefilename='extractedfluxes_autosave'):
        nframes = self.cleancube.shape[0]
        timingdata = self.timingdata
        all_fluxes = np.zeros((nframes, self.ncores))
        # all_fluxes_nonopt = np.zeros((nframes, self.ncores))
        t = time.time()
        for k in range(nframes):
            im = self.cleancube[k,:,:]
            for l in range(self.ncores):
                maskedim = im * self.all_weights[l, :, :]
                all_fluxes[k,l] = np.sum(maskedim)
                # maskedim = im * self.all_squaremasks[l, :, :]
                # all_fluxes_nonopt[k,l] = np.sum(maskedim)
                # # plt.clf()
                # # plt.imshow(maskedim)
                # # plt.colorbar()
                # # plt.pause(1)

            if k % 10000 == 0:
                eltime = time.time() - t
                print('Completed frame %d' % k + ' - %.3f seconds per 10000 frames' % eltime)
                t = time.time()

        self.all_fluxes = all_fluxes
        # self.all_fluxes_nonopt = all_fluxes_nonopt

        header_dict = dict(self.header)
        print('Saving to file '+savefilename+'.npz')
        np.savez(self.datapath+savefilename+'.npz', all_fluxes=all_fluxes, timingdata=timingdata, header=header_dict)
