import numpy as np
import time
import ctypes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import FliSdk_V2


class credcam:
    def __init__(self, camera_index=0, camera_id=None, verbose=False, cam_settings=None, cropdims=None,
                 darkpath='./', darkfile=None, buffersize_ims=1000):

        if cam_settings is None:
            self.dflt_settings = {'sensitivity': 'low',
                                  'bias mode': 'off',
                                  'flat mode': 'off',
                                  'badpixel mode': 'on',
                                  'fps': 600,#1e6, # 1e6 Sets to maximum
                                  'tint': 0.00005 # 1 Sets to maximum # 0.0002
                                  }
        else:
            self.dflt_settings = cam_settings

        self.cam_context = FliSdk_V2.Init()
        self.latest_im = None
        self.nims_lefttolog = 0
        self.nims_tolog = 1
        self.update_latestim = False
        self.syncdelay = None

        callback_fps = 0  # 0 for full speed

        self.darkpath = darkpath
        if darkfile is not None:
            self.dark = np.load(self.darkpath+darkfile+'.npy')
            if verbose:
                print('Using darkfile ' + self.darkpath+darkfile)
        else:
            self.dark = 0

        grabber_list = FliSdk_V2.DetectGrabbers(self.cam_context)
        camera_list = FliSdk_V2.DetectCameras(self.cam_context)

        if camera_id is not None:
            camera_index_l = [i for i, s in enumerate(camera_list) if camera_id in s]
            if len(camera_index_l) != 1:
                print("Error: couldn't find specified camera id")
            camera_index = camera_index_l[0]

        self.camera_index = camera_index

        num_cameras = len(camera_list)
        if verbose:
            print('%d cameras detected: ' % num_cameras)
            for k in range(num_cameras):
                print('%d: ' % k + camera_list[k])
            print('Using camera %d' % camera_index)

        errorval = FliSdk_V2.SetCamera(self.cam_context, camera_list[camera_index])
        FliSdk_V2.SetMode(self.cam_context, FliSdk_V2.Mode.Full)  # Enables grab and config
        errorval = FliSdk_V2.Update(self.cam_context)

        if cropdims is not None:
            self.send_command('set cropping columns %d-%d' % (cropdims[0], cropdims[1]), verbose=verbose)
            self.send_command('set cropping rows %d-%d' % (cropdims[2], cropdims[3]), verbose=verbose)
            self.send_command('set cropping on', verbose=verbose)
        else:
            self.send_command('set cropping off', verbose=verbose)
        self.camdims = FliSdk_V2.GetCurrentImageDimension(self.cam_context)

        # print('plcams.py camdims:')
        # print(self.camdims)
        # print('plcams.py cropdims:')
        # print(cropdims)


        buffersize_ims = buffersize_ims
        FliSdk_V2.SetBufferSizeInImages(self.cam_context, buffersize_ims)

        self.loggedims_cube = np.zeros((self.nims_tolog, self.camdims[1], self.camdims[0]), dtype=np.int16)
        self.loggedims_times_arr = np.zeros(self.nims_tolog)

        self.wrappedFunc = FliSdk_V2.CWRAPPER(self.newim_callbackfunc)
        FliSdk_V2.AddCallBackNewImage(self.cam_context, self.wrappedFunc, callback_fps, True, 0) #True

        self.external_trigger(enabled=False)
        FliSdk_V2.Start(self.cam_context)
        self.set_camera_defaults(verbose=verbose)


    def set_camera_defaults(self, verbose=True):
        print('Setting camera default settings:')
        self.send_command('set sensitivity ' + self.dflt_settings['sensitivity'], verbose=verbose)
        self.send_command('set bias ' + self.dflt_settings['bias mode'], verbose=verbose)
        self.send_command('set flat ' + self.dflt_settings['flat mode'], verbose=verbose)
        self.send_command('set badpixel ' + self.dflt_settings['badpixel mode'], verbose=verbose)
        self.send_command('set fps ' + str(self.dflt_settings['fps']), verbose=verbose)
        self.send_command('set tint ' + str(self.dflt_settings['tint']), verbose=verbose)

        if verbose:
            print(' ')
            print(self.send_command('fps'))
            print(self.send_command('tint'))
            

    def reset_buffer(self):
        FliSdk_V2.ResetBuffer(self.cam_context)


    def camera_start(self):
        FliSdk_V2.Start(self.cam_context)


    def camera_stop(self):
        FliSdk_V2.Stop(self.cam_context)


    def close(self):
        FliSdk_V2.Stop(self.cam_context)
        FliSdk_V2.Exit(self.cam_context)
        print('Closed context for camera %d' % self.camera_index)


    def external_trigger(self, enabled, syncdelay=None, verbose=False):
        if enabled:
            self.send_command('set extsynchro exposure internal', verbose=verbose)
            self.send_command('set extsynchro source external', verbose=verbose)
            self.send_command('set extsynchro on', verbose=verbose)
            self.send_command('set extsynchro polarity standard', verbose=verbose)
            if syncdelay is not None:
                self.syncdelay = syncdelay
                self.send_command('set syncdelay ' + str(syncdelay/1000), verbose=verbose)
            print('External synchronisation ENABLED')
        else:
            self.send_command('set extsynchro off', verbose=verbose)
            print('External synchronisation DISABLED')


    def set_tint(self, tint, verbose=False):
        self.send_command('set tint ' + str(tint), verbose=verbose)


    def take_dark(self, darkfile='default_dark', navs=1000, save=False):
        self.set_nims_tolog(navs)
        darkframe = self.get_n_images(return_ims=True, coadd=True)
        self.dark = darkframe
        print('New darkframe acquired')
        if save:
            print('Saving darkframe to '+self.darkpath+darkfile+'.npy')
            np.save(self.darkpath+darkfile+'.npy', darkframe)


    def load_dark(self, darkfile='default_dark'):
        darkframe = np.load(self.darkpath+darkfile+'.npy')
        self.dark = darkframe
        print('Loaded darkframe from '+self.darkpath+darkfile+'.npy')


    def send_command(self, commandstr, return_response=True, verbose=False):
        errorval, response = FliSdk_V2.FliSerialCamera.SendCommand(self.cam_context, commandstr)
        if verbose:
            print(commandstr)
            print(response)
        if return_response:
            return response


    def get_latest_image(self, return_im=True, waitfornewframe=True):
        if waitfornewframe:
            self.update_latestim = True
            while self.update_latestim:
                # self.goodtimer(0.1)
                pass
                # time.sleep(0.001)
        else:
            ## This doesn't support signed ints...
            # new_im = FliSdk_V2.GetRawImageAsNumpyArray(self.cam_context, -1)  # -1 gets most recent image

            image = FliSdk_V2.GetRawImage(self.cam_context, -1)
            ArrayType = ctypes.c_uint16 * self.camdims[0] * self.camdims[1]
            pa = ctypes.cast(image, ctypes.POINTER(ArrayType))
            buffer = np.ndarray((self.camdims[1], self.camdims[0]), dtype=np.int16, buffer=pa.contents)
            new_im = np.copy(buffer)
            self.latest_im = new_im

        self.loggedims_times_arr = [time.perf_counter()]
        if return_im:
            return self.latest_im


    def check_nims_buffer(self):
        n_buffer_ims = FliSdk_V2.GetBufferFilling(self.cam_context) + 1
        return n_buffer_ims


    def get_buffer_images(self, return_im=True, verbose=False):
        ## This doesn't support signed ints...
        # new_im = FliSdk_V2.GetRawImageAsNumpyArray(self.cam_context, -1)  # -1 gets most recent image

        n_buffer_ims = FliSdk_V2.GetBufferFilling(self.cam_context) + 1
        print('Num images in buffer: %.1f' % n_buffer_ims)

        self.loggedims_cube = np.zeros((n_buffer_ims, self.camdims[1], self.camdims[0]), dtype=np.int16)
        for k in range(n_buffer_ims):
            image = FliSdk_V2.GetRawImage(self.cam_context, k)
            ArrayType = ctypes.c_uint16 * self.camdims[0] * self.camdims[1]
            pa = ctypes.cast(image, ctypes.POINTER(ArrayType))
            buffer = np.ndarray((self.camdims[1], self.camdims[0]), dtype=np.int16, buffer=pa.contents)
            self.loggedims_cube[k,:,:] = np.copy(buffer)
        print('get_buffer_ims: loop done')
        if return_im:
            return self.loggedims_cube


    def newim_callbackfunc(self, image, ctx): #Use different method than numpy one to avoid dupes? E.g get straight
        # from a buffer?
        # new_im = FliSdk_V2.GetRawImageAsNumpyArray(self.cam_context, -1)  # -1 gets most recent image

        ArrayType = ctypes.c_uint16 * self.camdims[0] * self.camdims[1]
        pa = ctypes.cast(image, ctypes.POINTER(ArrayType))
        buffer = np.ndarray((self.camdims[1], self.camdims[0]), dtype=np.int16, buffer=pa.contents)

        new_im = np.copy(buffer)

        if self.update_latestim:
            self.latest_im = new_im ### TODO - double check this can't change im in loggedims_list (another copy?)
            self.update_latestim = False

        if self.nims_lefttolog > 0:
            self.loggedims_cube[self.nims_tolog-self.nims_lefttolog, :, :] = new_im
            self.loggedims_times_arr[self.nims_tolog-self.nims_lefttolog] = time.perf_counter()
            self.nims_lefttolog -= 1


    def get_n_images(self, blocking=True, return_ims=False, coadd=False, subtract_dark=False):
        self.nims_lefttolog = self.nims_tolog

        if blocking:
            while self.nims_lefttolog > 0:
                time.sleep(0.001)
            if subtract_dark:
                self.loggedims_cube = self.loggedims_cube - self.dark # TODO - does this slow it down, if coadding later?
            if return_ims:
                # loggedims_cube_copy = np.copy(self.loggedims_cube)
                # self.loggedims_cube = np.zeros((self.nims_tolog, self.camdims[1], self.camdims[0]), dtype=np.uint16)
                # return loggedims_cube_copy
                if coadd:
                    return np.mean(self.loggedims_cube, axis=0)
                else:
                    return self.loggedims_cube


    def set_nims_tolog(self, nims):
        self.nims_tolog = nims
        self.loggedims_cube = np.zeros((self.nims_tolog, self.camdims[1], self.camdims[0]), dtype=np.int16)
        self.loggedims_times_arr = np.zeros(self.nims_tolog)


    def goodtimer(self, time_ms):
        tm = time_ms/1000
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < tm:
            pass



