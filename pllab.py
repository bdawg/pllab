"""
Class to take sets of measurements with PL testbed
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
from plcams import credcam
from multiprocessing import shared_memory
import subprocess
import matplotlib
matplotlib.use('TkAgg')


# noinspection PyStringFormat
class pllab:
    def __init__(self, datadir='./', camstosave=('pl'), shm_mode=True, darkfiles=None, darkpath='./',
                 delays=(12,3), verbose=False, cam_settings=None, winparams=None, lutfile=None,
                 camdims=(640,512), cube_nims=1000):
        self.datadir = datadir
        self.darkfiles = darkfiles
        self.cam_syncdelay_ms = delays[0]
        self.extra_delay_ms = delays[1]
        self.wins = []
        self.verbose = verbose
        self.shm_mode = shm_mode
        self.camdims = camdims
        self.cube_nims = cube_nims
        self.all_imcubes = []

        # Define camera ids for each camera
        camids = {'pl': 'c86ca65',
                  'psf': '1de3043',
                  'refl': '1de3be41'}

        if shm_mode is False:
            print('Error: only shared memory mode is currently implemented')
            return
            # # Instantiate cameras
            # self.camobjs = []
            # for k in range(len(camstosave)):
            #     camid = camids[camstosave[k]]
            #     if cam_settings is None:
            #         cur_cam_settings = None
            #     else:
            #         cur_cam_settings = cam_settings[k]
            #     if darkfiles is None:
            #         darkfile = None
            #     else:
            #         darkfile = darkfiles[k]
            #     cam = credcam(camera_id=camid, darkfile=darkfile, darkpath=darkpath, verbose=verbose,
            #                   cam_settings=cur_cam_settings)
            #     self.camobjs.append(cam)
        else:
            print('Setting up cameras in shared memory mode')
            camindex = 0
            self.all_cam_indexes = []
            self.all_cam_commsl = []
            self.all_cam_imshm_obj = []
            self.all_subproc_camproc = []
            self.all_shmnames = []
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
                self.setup_shm_cameras(camid, camindex, cur_cam_settings, darkfile)
                camindex += 1


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


    def setup_shm_cameras(self, camid, camindex=0, cur_cam_settings=None, darkfile=None):
        # Set up shared memory for both shareable list (for communication) and image data array
        # Todo - either pass name of cam settings file via the commsl, or use dictionaries with ultradict

        sl_nitems = 20
        sl_maxitemlength = 100
        """
        commsl is structures as follows:
        [0]  - camera command send
        [1]  - camera response
        [2]  - max cube_nims
        [3]  - camera dim 0
        [4]  - camera dim 1
        [5]  - camera process ready
        [6]  - current cube_nims, if < max cube_nims
        
        [10] - cam acquire. Set to 1 to acquire cube, and camprocess will set to 0 when acq complete.
        """
        cam_commsl_shmname = 'cam%d_commsl' % camindex
        cam_imshm_shmname = 'cam%d_imshm' % camindex
        self.all_shmnames.append(cam_commsl_shmname)
        self.all_shmnames.append(cam_imshm_shmname)

        try: # To remove shms if not previously unlinked:
            shared_memory.SharedMemory(name=cam_commsl_shmname).unlink()
            shared_memory.SharedMemory(name=cam_imshm_shmname).unlink()
        except:
            pass
        time.sleep(1)

        cam_commsl = shared_memory.ShareableList([' ' * sl_maxitemlength] * sl_nitems, name=cam_commsl_shmname)
        for k in range(sl_nitems): # Once memory allocated, initialise with Nones
            cam_commsl[k] = None

        init_cube = np.zeros((self.cube_nims, self.camdims[1], self.camdims[0]), dtype=np.int16)
        cam_imshm_obj = shared_memory.SharedMemory(name=cam_imshm_shmname, create=True, size=init_cube.nbytes)
        cam_imshm = np.ndarray(init_cube.shape, dtype=init_cube.dtype, buffer=cam_imshm_obj.buf)
        cam_imshm[:] = np.copy(init_cube)
        del init_cube

        cam_commsl[2] = self.cube_nims
        cam_commsl[3] = self.camdims[0]
        cam_commsl[4] = self.camdims[1]
        cam_commsl[5] = 0
        cam_commsl[10] = 0

        # Set up camera processes
        if self.verbose:
            verbose_state = '1'
        else:
            verbose_state = '0'
        args0 = ['python', 'plcam_camprocess.py', camid, cam_commsl_shmname, cam_imshm_shmname, verbose_state]
        subproc_camproc = subprocess.Popen(args0)
        while cam_commsl[5] != 1:
            if self.verbose:
                print('Waiting for cam setup to complete...')
            time.sleep(5)
        self.all_cam_commsl.append(cam_commsl)
        # self.all_cam_imshm.append(cam_imshm)
        self.all_cam_imshm_obj.append(cam_imshm_obj)
        self.all_subproc_camproc.append(subproc_camproc)
        self.all_cam_indexes.append(camindex)
        print('Camera ' + camid + ' ready.')


    def unlink_allshm(self):
        for shmname in self.all_shmnames:
            shared_memory.SharedMemory(name=shmname).unlink()


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


    def run_measurements_shm(self, return_data=False, current_cube_nims=None, truncate_zeros=True):
        if current_cube_nims is not None:
            cube_nims = current_cube_nims
            for sl in self.all_cam_commsl:
                sl[6] = current_cube_nims
        else:
            cube_nims = self.cube_nims
        n_slmfrms = self.all_slmims.shape[0]
        if cube_nims % n_slmfrms != 0:
            print('Error: cube_nims must be a multiple of number of input slm frames')
            return
        nloops = int(cube_nims / n_slmfrms)
        wait_time_ms = self.cam_syncdelay_ms + self.extra_delay_ms
        self.all_imcubes = []

        # Set command to start acquiring
        for sl in self.all_cam_commsl:
            sl[10] = 1
        time.sleep(0.1) # Allow for polling rate of camprocesses to sl

        # Take data
        starttime = time.time()
        print('Beginning measurement set for %d loops of SLM image data file ' % nloops + self.slmims_filename)
        count = 0
        for k in range(nloops):
            for l in range(n_slmfrms):
                if count % 100 == 0:
                    print('Acquiring measurement %d' % count)
                slmim = self.all_slmims[l,:, :]
                self.slm.slmwrite(slmim, showplot=False, skip_readycheck=True)
                count += 1
                self.goodtimer(wait_time_ms)

        waiting = True
        while waiting:
            aqstatsum = 0
            for sl in self.all_cam_commsl:
                aqstatsum += sl[10]
            if aqstatsum == 0:
                waiting = False
            if self.verbose:
                print('Waiting for camprocesses to finish acquiring...')
            time.sleep(0.5)
        print('Acquisition complete - elapsed time %.2f seconds' % (time.time() - starttime))

        for k in self.all_cam_indexes:
            cam_imshm_obj = self.all_cam_imshm_obj[k]
            cam_imshm = np.ndarray((self.cube_nims, self.camdims[1], self.camdims[0]), dtype=np.int16,
                                   buffer=cam_imshm_obj.buf)
            data = np.copy(cam_imshm)
            if truncate_zeros and (cube_nims < self.cube_nims):
                data = np.copy(cam_imshm)
                self.all_imcubes.append(data[:cube_nims, :, :])
            else:
                self.all_imcubes.append(np.copy(cam_imshm))

        if return_data:
            return self.all_imcubes


    def send_shm_camcommand(self, cam_index, cmd_string, return_response=False):
        # commsl is structures as follows:
        # [0]  - camera command send
        # [1]  - camera response
        sl = self.all_cam_commsl[cam_index]
        sl[0] = cmd_string
        while sl[1] is None:
            time.sleep(0.01)
        resp = sl[1]
        sl[1] = None
        sl[0] = None
        if self.verbose:
            print('Sent to camera_index %d command: ' % cam_index + cmd_string)
            print('Received response: ' + resp)
        if return_response:
            return resp


    # def run_measurements(self, nloops=1, subrange=None, return_data=False):
    #     n_ims = self.all_slmims.shape[0] * nloops
    #     n_cams = len(self.camobjs)
    #     self.allcams_camims = []
    #     self.allcams_imtimes = []
    #
    #     # Preallocate image cubes
    #     for c in range(n_cams):
    #         if len(self.wins) > 0:
    #             docrop = True
    #             win = self.wins[c]
    #             camdim = [win[1]-win[0], win[3]-win[2]]
    #         else:
    #             docrop = False
    #             camdim = [self.camobjs[c].camdims[0], self.camobjs[c].camdims[1]]
    #         self.allcams_camims.append(np.zeros((n_ims, camdim[1], camdim[0]), dtype=np.int16))
    #         self.allcams_imtimes.append(np.zeros(n_ims))
    #
    #     # Take data
    #     print('Beginning measurement set for %d loops of SLM image data file ' % nloops + self.slmims_filename)
    #     count = 0
    #     for lp in range(nloops):
    #         for k in range(self.all_slmims.shape[0]):
    #             if count % 100 == 0:
    #                 print('Acquiring measurement %d' % count)
    #
    #             slmim = self.all_slmims[k, :, :]
    #             self.slm.slmwrite(slmim, showplot=False)
    #
    #             self.goodtimer(self.latency)
    #
    #             for c in range(n_cams):
    #                 # self.goodtimer(self.latency)
    #
    #                 camim = self.camobjs[c].get_latest_image(waitfornewframe=False, return_im=True)
    #                 # if docrop:
    #                 #     win = self.wins[c]
    #                 #     camim = camim[win[0]:win[1], win[2]:win[3]]
    #                 self.allcams_camims[c][count, :, :] = camim
    #                 self.allcams_imtimes[c][count] = self.camobjs[c].loggedims_times_arr[0]
    #
    #             count += 1
    #
    #     dtimes0 = np.diff(self.allcams_imtimes[0])*1000
    #
    #     print('Acquisition complete.')
    #     print('Time per frame %.2f +- %.2f ms' % (np.mean(dtimes0), np.std(dtimes0)))
    #     if return_data:
    #         return self.allcams_camims


    def goodtimer(self, time_ms):
        tm = time_ms/1000
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < tm:
            pass


    def imfluxes(self, camnum):
        flux = np.mean(self.allcams_camims[camnum], (1,2))
        return flux


    def plot_imfluxes(self, window=False, winparams=None, fignum=1):
        if winparams is not None:
            wins = []
            for winparam in winparams:
                wsz = winparam[2]
                win = (winparam[0] - wsz // 2, winparam[0] + wsz // 2, winparam[1] - wsz // 2,
                       winparam[1] + wsz // 2)
                wins.append(win)
        else:
            wins = self.wins
        plt.figure(fignum)
        plt.clf()
        nplots = len(self.all_cam_indexes)
        for k in self.all_cam_indexes:
            if window:
                cube = self.all_imcubes[k][:, wins[k][0]:wins[k][1], wins[k][2]:wins[k][3]]
            else:
                cube = self.all_imcubes[k]
            cubefluxes = np.sum(cube, axis=(1, 2))
            plt.subplot(nplots, 1, k+1)
            plt.plot(cubefluxes)
            plt.title('Camera %d fluxes' % k)


    def makestripecube(self, n_slmims=10, period=50, ampl_range=(0, 60), type='square', savefile='',
                       set_as_current=True, showplot=False, return_cube=False):
        amplvals = np.linspace(ampl_range[0], ampl_range[1], n_slmims)
        slmim_dtype = 'int8'
        all_slmims = np.zeros((n_slmims, self.slm.slmdims[0], self.slm.slmdims[1]), dtype=slmim_dtype)
        all_slmim_params = np.zeros((n_slmims, 2))
        for k in range(n_slmims):
            period = period
            amplitude = amplvals[k]
            slm_im =  self.slm.makestripes(period=period, ampl=amplitude, phi=0, type=type, showplot=showplot,
                                           sendtoslm=False, return_im=True)
            all_slmims[k, :, :] = slm_im
            all_slmim_params[k, :] = [period, amplitude]

        print('Generated %d SLM frames' % n_slmims)
        if len(savefile) > 0:
            np.savez_compressed(self.datadir+savefile, all_slmims=all_slmims, all_slmim_params=all_slmim_params)
            print('Saved SLM cube to ' + savefile)

        if set_as_current:
            self.all_slmims = all_slmims
            self.all_slmim_params = all_slmim_params
            self.slmims_filename = savefile

        if return_cube:
            return all_slmims


    def savedata(self, filename, cam_subset=None, save_format='npz'):
        if cam_subset is None:
            cam_subset = self.all_cam_indexes
        savesubstr = ''
        for ind in cam_subset:
            savesubstr = savesubstr + 'imcube_cam%d=self.all_imcubes[%d], ' % (ind, ind)
        savestr = 'np.savez(self.datadir+filename, ' + savesubstr + \
                  'all_slmim_params=self.all_slmim_params, slmims_filename=self.slmims_filename)'
        print('Saving data to ' + filename)
        exec(savestr)
        print('Saving done.')



















