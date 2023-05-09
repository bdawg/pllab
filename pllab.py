"""
Class to take sets of measurements with PL testbed
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')


# noinspection PyStringFormat
class pllab:
    def __init__(self, datadir='./', camstosave=('pl'), darkfiles=None, darkpath='./',
                 latency=10.0, verbose=False, cam_settings=None, winparams=None, lutfile=None):
        self.datadir = datadir
        self.darkfiles = darkfiles
        self.latency = latency
        self.wins = []

        # Define camera ids for each camera
        camids = {'pl': 'c86ca65',
                  'psf': '1de3043',
                  'refl': '1de3be41'}

        # Instantiate cameras
        self.camobjs = []
        for k in range(len(camstosave)):
            camid = camids[camstosave[k]]
            if cam_settings is None:
                cur_cam_settings = None
            else:
                cur_cam_settings = cam_settings[k]
            if darkfiles is None:
                darkfile = None
            else:
                darkfile = darkfiles[k]
            cam = credcam(camera_id=camid, darkfile=darkfile, darkpath=darkpath, verbose=verbose,
                          cam_settings=cur_cam_settings)
            self.camobjs.append(cam)

        # Instantiate SLM
        self.slm = plslm(lutfile=lutfile)

        # Set up image windowing (but ideally do this in-camera)
        # imwins contains (x, y, winsize) for each camera
        if winparams is not None:
            for winparam in winparams:
                wsz = winparam[2]
                win = (winparam[0] - wsz // 2, winparam[0] + wsz // 2, winparam[1] - wsz // 2,
                       winparam[1] + wsz // 2)
                self.wins.append(win)


    def load_slmims(self, slmims_filename, slmims_path=''):
        slmimdataf = np.load(slmims_path + slmims_filename)
        slmims = slmimdataf['all_slmims']
        if type(slmims[0,0,0]) is not np.int8:
            print('Error: input SLM cube not int8')
            return
        self.all_slmims = slmims
        self.all_slmim_params = slmimdataf['all_slmim_params']
        self.slmims_filename = slmims_filename
        print('Loaded SLM image data file '+slmims_path + slmims_filename)


    def run_measurements(self, nloops=1, subrange=None, return_data=False):
        n_ims = self.all_slmims.shape[0] * nloops
        n_cams = len(self.camobjs)
        self.allcams_camims = []
        self.allcams_imtimes = []

        # Preallocate image cubes
        for c in range(n_cams):
            if len(self.wins) > 0:
                docrop = True
                win = self.wins[c]
                camdim = [win[1]-win[0], win[3]-win[2]]
            else:
                docrop = False
                camdim = [self.camobjs[c].camdims[0], self.camobjs[c].camdims[1]]
            self.allcams_camims.append(np.zeros((n_ims, camdim[1], camdim[0]), dtype=np.int16))
            self.allcams_imtimes.append(np.zeros(n_ims))

        # Take data
        print('Beginning measurement set for %d loops of SLM image data file ' % nloops + self.slmims_filename)
        count = 0
        for lp in range(nloops):
            for k in range(self.all_slmims.shape[0]):
                if count % 100 == 0:
                    print('Acquiring measurement %d' % count)

                slmim = self.all_slmims[k, :, :]
                self.slm.slmwrite(slmim, showplot=False)

                self.goodtimer(self.latency)

                for c in range(n_cams):
                    # self.goodtimer(self.latency)

                    camim = self.camobjs[c].get_latest_image(waitfornewframe=False, return_im=True)
                    # if docrop:
                    #     win = self.wins[c]
                    #     camim = camim[win[0]:win[1], win[2]:win[3]]
                    self.allcams_camims[c][count, :, :] = camim
                    self.allcams_imtimes[c][count] = self.camobjs[c].loggedims_times_arr[0]

                count += 1

        dtimes0 = np.diff(self.allcams_imtimes[0])*1000

        print('Acquisition complete.')
        print('Time per frame %.2f +- %.2f ms' % (np.mean(dtimes0), np.std(dtimes0)))
        if return_data:
            return self.allcams_camims


    def goodtimer(self, time_ms):
        tm = time_ms/1000
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < tm:
            pass


    def imfluxes(self, camnum):
        flux = np.mean(self.allcams_camims[camnum], (1,2))
        return flux