import numpy as np
import time
import ctypes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import FliSdk_V2


class credcam:
    def __init__(self, camera_index=0, verbose=False, dflt_settings=None, darkpath='./darks/',
                 darkfile='default_dark'):

        if dflt_settings is None:
            self.dflt_settings = {'sensitivity': 'low',
                                  'bias mode': 'off',
                                  'flat mode': 'off',
                                  'badpixel mode': 'on',
                                  'fps': 600,#1e6, # 1e6 Sets to maximum
                                  'tint': 0.00005 # 1 Sets to maximum # 0.0002
                                  }
        else:
            self.dflt_settings = dflt_settings

        self.cam_context = FliSdk_V2.Init()
        self.camera_index = camera_index
        self_latest_im = None
        self.nims_lefttolog = 0
        self.nims_tolog = 1
        self.update_latestim = False

        callback_fps = 0  # 0 for full speed

        self.darkpath = darkpath
        try:
            self.dark = np.load(self.darkpath+darkfile+'.npy')
        except:
            print('Could not load darkfile '+ self.darkpath+darkfile+'.npy, setting to 0')
            self.dark = 0

        grabber_list = FliSdk_V2.DetectGrabbers(self.cam_context)
        camera_list = FliSdk_V2.DetectCameras(self.cam_context)

        num_cameras = len(camera_list)
        if verbose:
            print('%d cameras detected: ' % num_cameras)
            for k in range(num_cameras):
                print('%d: ' % k + camera_list[k])
            print('Using camera %d' % camera_index)

        errorval = FliSdk_V2.SetCamera(self.cam_context, camera_list[camera_index])
        FliSdk_V2.SetMode(self.cam_context, FliSdk_V2.Mode.Full)  # Enables grab and config
        errorval = FliSdk_V2.Update(self.cam_context)
        self.camdims = FliSdk_V2.GetCurrentImageDimension(self.cam_context)

        # TODO
        buffersize_ims = 1000
        FliSdk_V2.SetBufferSizeInImages(self.cam_context, buffersize_ims)

        self.loggedims_cube = np.zeros((self.nims_tolog, self.camdims[1], self.camdims[0]), dtype=np.int16)
        self.loggedims_times_arr = np.zeros(self.nims_tolog)

        self.wrappedFunc = FliSdk_V2.CWRAPPER(self.newim_callbackfunc)
        FliSdk_V2.AddCallBackNewImage(self.cam_context, self.wrappedFunc, callback_fps, False, 0)

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


    def close(self):
        FliSdk_V2.Stop(self.cam_context)
        FliSdk_V2.Exit(self.cam_context)
        print('Closed context for camera %d' % self.camera_index)


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


    def send_command(self, commandstr, return_response = True, verbose = False):
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
                pass
                # time.sleep(0.001)
        else:
            # This doesn't support signed ints...
            new_im = FliSdk_V2.GetRawImageAsNumpyArray(self.cam_context, -1)  # -1 gets most recent image
            # ArrayType = ctypes.c_uint16 * self.camdims[0] * self.camdims[1]
            # pa = ctypes.cast(image, ctypes.POINTER(ArrayType))
            # buffer = np.ndarray((self.camdims[1], self.camdims[0]), dtype=np.int16, buffer=pa.contents)
            # new_im = np.copy(buffer)
            self.latest_im = new_im

        self.loggedims_times_arr = [time.perf_counter()]
        if return_im:
            return self.latest_im


    def newim_callbackfunc(self, image, ctx): #Use different method than numpy one to avoid dupes? E.g get straight from a buffer?
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






