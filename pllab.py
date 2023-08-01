"""
Class to take sets of measurements with PL testbed
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import os
from plslm import plslm
from plcams import credcam
from multiprocessing import shared_memory
import subprocess
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')


# noinspection PyStringFormat
class pllab:
    def __init__(self, datadir='./', camstosave=('pl'), shm_mode=True, darkfile=None, darkpath='./',
                 delays=(12,3), verbose=False, cam_settings=None, winparams=None, lutfile=None,
                 camdims=(640,512), cropdims=None, cube_nims=1000, winparams_fluxsum=None):
        self.datadir = datadir
        self.darkfile = darkfile
        self.darkpath = darkpath
        self.cam_syncdelay_ms = delays[0]
        self.extra_delay_ms = delays[1]
        self.wins = []
        self.verbose = verbose
        self.shm_mode = shm_mode
        # self.camdims = camdims
        self.cube_nims = cube_nims
        self.cropdims = cropdims
        self.all_imcubes = []
        self.darkframes = []
        self.slmims_filename = ''

        # Define camera ids for each camera
        camids = {'pl': 'c86ca65',
                  'psf': '1de3043',
                  'refl': '1de3be41'}

        if winparams_fluxsum is None:
            # self.winparams_fluxsum = [[257, 340, 42],
            #                           [160, 282, 120]]
            self.winparams_fluxsum = [[145, 141, 42], #Crop mode
                                      [88, 77, 120]]
        else:
            self.winparams_fluxsum = winparams_fluxsum

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
            self.camdims = []
            for k in range(len(camstosave)):
                camid = camids[camstosave[k]]
                if cam_settings is None:
                    cur_cam_settings = None
                else:
                    cur_cam_settings = cam_settings[k]

                if self.cropdims is not None:
                    ncols = self.cropdims[camindex][1] - self.cropdims[camindex][0] + 1
                    nrows = self.cropdims[camindex][3] - self.cropdims[camindex][2] + 1
                    self.camdims.append([ncols, nrows])
                else:
                    self.camdims.append(camdims)

                self.setup_shm_cameras(camid, camindex, cur_cam_settings)#, darkfile)
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

        if self.darkfile is not None:
            try:
                self.load_darks()
            except:
                print('Warning: could not load darkfile.')


    def setup_shm_cameras(self, camid, camindex=0, cur_cam_settings=None): #, darkfile=None):
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
        [11] to [14] - cropdims (or None for full frame)
        
        cropdims are defined as [FirstColumn, LastColumn, FirstRow, LastRow]. 
        Columns must be in steps of 32, rows in steps of 4.
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

        init_cube = np.zeros((self.cube_nims, self.camdims[camindex][1], self.camdims[camindex][0]), dtype=np.int16)
        cam_imshm_obj = shared_memory.SharedMemory(name=cam_imshm_shmname, create=True, size=init_cube.nbytes)
        cam_imshm = np.ndarray(init_cube.shape, dtype=init_cube.dtype, buffer=cam_imshm_obj.buf)
        cam_imshm[:] = np.copy(init_cube)
        del init_cube

        cam_commsl[2] = self.cube_nims
        cam_commsl[3] = self.camdims[camindex][0]
        cam_commsl[4] = self.camdims[camindex][1]
        cam_commsl[5] = 0
        cam_commsl[10] = 0

        if self.cropdims is not None:
            # cropdims are defined as [FirstColumn, LastColumn, FirstRow, LastRow].
            # Columns must be in steps of 32, rows in steps of 4.
            cropdims = self.cropdims[camindex]
            cam_commsl[11] = cropdims[0]
            cam_commsl[12] = cropdims[1]
            cam_commsl[13] = cropdims[2]
            cam_commsl[14] = cropdims[3]

        # ##### DEBUG #####
        # print(' ')
        # print('Debug from pllab.setup_shm_cameras()')
        # print('Camera index %d' % camindex)
        # print('Camera ' + camid)
        # print([cam_commsl[2], cam_commsl[3], cam_commsl[4], cam_commsl[5], cam_commsl[10]])
        # #####

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


    def load_slmims(self, slmims_filename, slmims_path='', slmim_array_name='all_slmims',
                    slmim_param_name='all_slmim_params', insert_as_subimage=False, slmloc=None):
        print('Loading SLM data...')
        slmimdataf = np.load(slmims_path + slmims_filename, allow_pickle=True)
        slmims = slmimdataf[slmim_array_name]
        if type(slmims[0,0,0]) is not np.uint8:
            print('Warning: input SLM cube not uint8, converting.')
            slmims = slmims.astype('uint8')
            # return

        if insert_as_subimage:
            # slmloc=np.array([slm_centre[0], slm_centre[1], slm_rad])
            if slmloc is None:
                slmloc = slmimdataf['slmloc']
            slm_centre = slmloc[:2]
            print('Using SLM centre: ')
            print(slm_centre)
            slm_rad = slmloc[2]
            num_samps = slmims.shape[0]
            full_slmims = np.ones((num_samps, self.slm.slmdims[0], self.slm.slmdims[1]), dtype='uint8') * 127
            full_slmims[:, slm_centre[0] - slm_rad:slm_centre[0] + slm_rad, \
                slm_centre[1] - slm_rad:slm_centre[1] + slm_rad] = slmims
            self.all_slmims = full_slmims
        else:
            self.all_slmims = slmims


        try:
            self.all_slmim_params = slmimdataf[slmim_param_name]
        except:
            print('Warning: no slmim params array found')
            self.all_slmim_params = None
        self.slmims_filename = slmims_filename
        print('Loaded SLM image data file '+slmims_path + slmims_filename)


    def run_measurements_shm(self, return_data=False, current_cube_nims=None, truncate_zeros=True,
                             plot_final=False, plot_whileacq=False):
        if current_cube_nims is not None:
            cube_nims = current_cube_nims
            for sl in self.all_cam_commsl:
                sl[6] = current_cube_nims
        else:
            cube_nims = self.cube_nims
        n_slmfrms = self.all_slmims.shape[0]
        if (cube_nims % n_slmfrms != 0):
            print('Error: cube_nims must be a multiple of number of input slm frames')
            return
        nloops = int(cube_nims / n_slmfrms)
        wait_time_ms = self.cam_syncdelay_ms + self.extra_delay_ms
        self.all_imcubes = []

        # Set command to start acquiring
        for sl in self.all_cam_commsl:
            sl[10] = 1
        time.sleep(0.5) # Allow for polling rate of camprocesses to sl #TODO was 0.1

        # Take data
        starttime = time.time()
        print('Beginning measurement set for %d loops of SLM image data file ' % nloops + self.slmims_filename)
        count = 0
        for k in range(nloops):
            for l in range(n_slmfrms):
                slmim = self.all_slmims[l,:, :]
                if count % 100 == 0:
                    print('Acquiring measurement %d' % count)
                    if plot_whileacq:
                        self.plot_curr(slmim, count)
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

        for camindex in self.all_cam_indexes:
            cam_imshm_obj = self.all_cam_imshm_obj[camindex]
            cam_imshm = np.ndarray((self.cube_nims, self.camdims[camindex][1], self.camdims[camindex][0]),
                                   dtype=np.int16, buffer=cam_imshm_obj.buf)
            data = np.copy(cam_imshm) # TODO fix
            if truncate_zeros and (cube_nims < self.cube_nims):
                data = np.copy(cam_imshm)
                self.all_imcubes.append(data[:cube_nims, :, :])
            else:
                self.all_imcubes.append(np.copy(cam_imshm))

        if plot_final:
            self.show_ims()

        if return_data:
            return self.all_imcubes


    def take_darks(self, darkfile='default_dark', navs=1000, save=False):
        self.darkframes = []
        cube_nims = navs
        for sl in self.all_cam_commsl:
            sl[6] = cube_nims
        wait_time_ms = self.cam_syncdelay_ms + self.extra_delay_ms

        for sl in self.all_cam_commsl:
            sl[10] = 1
        time.sleep(0.5) # Allow for polling rate of camprocesses to sl #TODO was 0.1

        slmim = np.zeros((self.slm.slmdims[0], self.slm.slmdims[1]), dtype='uint8')
        starttime = time.time()
        count = 0
        for k in range(cube_nims):
            if count % 100 == 0:
                print('Acquiring measurement %d' % count)
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

        for camindex in self.all_cam_indexes:
            cam_imshm_obj = self.all_cam_imshm_obj[camindex]
            cam_imshm = np.ndarray((cube_nims, self.camdims[camindex][1], self.camdims[camindex][0]),
                                   dtype=np.int16, buffer=cam_imshm_obj.buf)
            imcube = np.copy(cam_imshm)
            darkframe = np.mean(imcube,0)
            self.darkframes.append((darkframe))

        if save:
            if not os.path.exists(self.darkpath + darkfile):
                print('Saving to ' + self.darkpath + darkfile)
                np.savez(self.darkpath+darkfile, darkframes=self.darkframes,
                         datestr=datetime.now().isoformat())
            else:
                print('Error: file already exist, dark NOT saved.')


    def load_darks(self, darkfile=None):
        if darkfile is not None:
            self.darkfile = darkfile
        df = np.load(self.darkpath + self.darkfile, allow_pickle=True)
        self.darkframes = df['darkframes']
        print('Loaded darkframe ' + self.darkfile + ' with date ' + df['datestr'].item())


    def show_ims(self, imagedata=None, ncams=2, fignum=0, zero_firstrow=True,
                 winparams=None):
        if imagedata is None:
            imagedata = self.all_imcubes
        else:
            if not isinstance(imagedata, list):
                imagedata = np.expand_dims(imagedata, 0)
                imagedata = [imagedata]
                ncams = 1
        if winparams is not None:
            wins = []
            for winparam in winparams:
                wsz = winparam[2]
                win = (winparam[0] - wsz // 2, winparam[0] + wsz // 2, winparam[1] - wsz // 2,
                       winparam[1] + wsz // 2)
                wins.append(win)
        plt.figure(fignum)
        plt.clf()
        for k in range(ncams):
            plt.subplot(ncams+1, 1, k+1)
            im = imagedata[k][-1,:,:]
            if len(self.darkframes) > 0:
                im = im - self.darkframes[k]
            if zero_firstrow:
                im[0,:] = 0
            if winparams is not None:
                im = im[wins[k][0]:wins[k][1], wins[k][2]:wins[k][3]]
            plt.imshow(im)
        plt.subplot(ncams+1, 1, ncams+1)
        plt.imshow(self.slm.nextim)
        plt.tight_layout()
        plt.pause(0.001)


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


    def imfluxes(self,  window=False, winparams=None):
        if winparams is None:
            winparams = self.winparams_fluxsum
        wins = []
        for winparam in winparams:
            wsz = winparam[2]
            win = (winparam[0] - wsz // 2, winparam[0] + wsz // 2, winparam[1] - wsz // 2,
                   winparam[1] + wsz // 2)
            wins.append(win)
        all_fluxes = []
        for k in self.all_cam_indexes:
            cube = self.all_imcubes[k]
            if len(self.darkframes) > 0:
                cube = cube - self.darkframes[k]
            if window:
                cube = cube[:, wins[k][0]:wins[k][1], wins[k][2]:wins[k][3]]
            cubefluxes = np.sum(cube, axis=(1, 2))
            all_fluxes.append(cubefluxes)
        return all_fluxes


    def plot_imfluxes(self, window=False, winparams=None, fignum=1):
        all_fluxes = self.imfluxes(window=window, winparams=winparams)

        plt.figure(fignum)
        plt.clf()
        nplots = len(self.all_cam_indexes)
        for k in self.all_cam_indexes:
            plt.subplot(nplots, 1, k+1)
            plt.plot(all_fluxes[k])
            plt.title('Camera %d fluxes' % k)


    def makestripecube(self, n_slmims=10, period=50, ampl_range=(0, 60), type='square', savefile='',
                       set_as_current=True, showplot=False, return_cube=False):
        amplvals = np.linspace(ampl_range[0], ampl_range[1], n_slmims)
        slmim_dtype = 'uint8'
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


    def savedata(self, filename, cam_subset=None, savedir=None):
        if os.path.exists(self.datadir+filename):
            print('Error: file already exists, data NOT saved')
            return
        if cam_subset is None:
            cam_subset = self.all_cam_indexes
        if savedir is None:
            savedir = self.datadir
        savesubstr = ''
        for ind in cam_subset:
            savesubstr = savesubstr + 'imcube_cam%d=self.all_imcubes[%d], ' % (ind, ind)
        savestr = 'np.savez(savedir+filename, ' + savesubstr + \
                  'all_slmim_params=self.all_slmim_params, slmims_filename=self.slmims_filename, ' + \
                  'darkframes=self.darkframes, darkfile=self.darkfile)'
        print('Saving data to ' + filename)
        exec(savestr)
        print('Saving done.')


    def plot_curr(self, slmim, count, fignum=2, figsize=(4,10)):
        plt.figure(fignum)
        plt.clf()
        plt.imshow(slmim)
        plt.colorbar()
        plt.title('SLM image')
        plt.subplot(311)
        plt.imshow(slmim)
        plt.colorbar()
        plt.title('SLM image for frame %d' % count)
        plt.pause(0.001)


    def slmposnscan(self, num_lin_posns=32, ksz=32, meas_range=None, period=10, ampl=100, showplot=True,
                    plot_whileacq=False, circle_centre=None):

        print('Generating SLM patterns...')
        strfrm = self.slm.makestripes(period=period, ampl=ampl, return_im=True)
        slmdim = self.slm.slmdims[0]
        if meas_range is None:
            meas_posns = [np.linspace(0, slmdim, num_lin_posns).astype(int),
                          np.linspace(0, slmdim, num_lin_posns).astype(int)]
        else:
            meas_posns = [np.linspace(meas_range[0], meas_range[1], num_lin_posns).astype(int),
                          np.linspace(meas_range[2], meas_range[3], num_lin_posns).astype(int)]
        # hw = int((meas_range[1]-meas_range[0]) / num_posns / 2)
        n_meas = num_lin_posns**2

        # circle_centre = [600, 400]
        # meas_range = [0, 100]
        # num_lin_posns = 10
        if circle_centre is not None:
            Y, X = np.mgrid[-slmdim / 2:slmdim / 2, -slmdim / 2:slmdim / 2]
            # Y, X = np.mgrid[0:slmdim, 0:slmdim]
            X1 = X - circle_centre[1] + slmdim/2
            Y1 = Y - circle_centre[0] + slmdim/2
            # X1 = X + circle_centre[1]
            # Y1 = Y + circle_centre[0]
            R = np.sqrt(X1 ** 2 + Y1 ** 2)
            meas_posns[1] = [0]
            n_meas = num_lin_posns

        all_slmims = []
        for y in meas_posns[0]:
            for x in meas_posns[1]:
                mask = np.zeros((slmdim, slmdim))
                if circle_centre is None:
                    mask[y-ksz//2:y+ksz//2, x-ksz//2:x+ksz//2] = 1
                else:
                    mask[R <= y] = 1
                cur_slmim = strfrm * mask
                all_slmims.append(cur_slmim.astype('uint8'))
        self.all_slmims = np.array(all_slmims)
        print('...done.')

        self.run_measurements_shm(current_cube_nims=int(n_meas), plot_whileacq=plot_whileacq)
        all_fluxes = self.imfluxes(window=True)

        dflux_psf = all_fluxes[0][:] - all_fluxes[0][0]
        df_im = dflux_psf.reshape((len(meas_posns[0]),len(meas_posns[1])))
        if showplot:
            plt.clf()
            plt.imshow(df_im, extent=[meas_posns[1][0], meas_posns[1][-1], meas_posns[0][-1], meas_posns[0][0]])
        return df_im






















