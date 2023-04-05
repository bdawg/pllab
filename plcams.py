import numpy as np
import sys
import time
import ctypes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import FliSdk_V2


class credcam:
    def __init__(self, camera_index=0, verbose=False):
        self.cam_context = FliSdk_V2.Init()
        self.camera_index = camera_index
        self_latest_im = None
        # self.loggedims_list = []
        # self.loggedims_times = []
        self.do_logimages = False
        self.nims_lefttolog = 0
        self.nims_tolog = 1

        callback_fps = 0  # 0 for full speed

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

        self.loggedims_cube = np.zeros((self.nims_tolog, self.camdims[1], self.camdims[0]), dtype=np.uint16)
        self.loggedims_times_arr = np.zeros(self.nims_tolog)

        self.wrappedFunc = FliSdk_V2.CWRAPPER(self.newim_callbackfunc)
        FliSdk_V2.AddCallBackNewImage(self.cam_context, self.wrappedFunc, callback_fps, False, 0)

        FliSdk_V2.Start(self.cam_context)
        self.set_camera_defaults(verbose=verbose)


    ### Can't get this to work... unnecessary?
    # def stop_callbacks(self):
    #     FliSdk_V2.RemoveCallBackNewImage(self.cam_context, 0)


    def set_camera_defaults(self, verbose=True):
        dflt_settings = {'sensitivity': 'low',
                              'bias mode': 'off',
                              'flat mode': 'off',
                              'badpixel mode': 'on',
                              'fps': 100,#1e6, # 1e6 Sets to maximum
                              'tint': 0.0002 # 1 Sets to maximum
                              }

        print('Setting camera default settings:')
        self.send_command('set sensitivity ' + dflt_settings['sensitivity'], verbose=verbose)
        self.send_command('set bias ' + dflt_settings['bias mode'], verbose=verbose)
        self.send_command('set flat ' + dflt_settings['flat mode'], verbose=verbose)
        self.send_command('set badpixel ' + dflt_settings['badpixel mode'], verbose=verbose)
        self.send_command('set fps ' + str(dflt_settings['fps']), verbose=verbose)
        self.send_command('set tint ' + str(dflt_settings['tint']), verbose=verbose)

        if verbose:
            print(' ')
            print(self.send_command('fps'))
            print(self.send_command('tint'))


    def close(self):
        FliSdk_V2.Stop(self.cam_context)
        FliSdk_V2.Exit(self.cam_context)
        print('Closed context for camera %d' % self.camera_index)


    def send_command(self, commandstr, return_response = True, verbose = False):
        errorval, response = FliSdk_V2.FliSerialCamera.SendCommand(self.cam_context, commandstr)
        if verbose:
            print(commandstr)
            print(response)
        if return_response:
            return response


    def get_latest_image(self, return_im=True):
        new_im = FliSdk_V2.GetRawImageAsNumpyArray(self.cam_context, -1)  # -1 gets most recent image
        self.latest_im = new_im

        if return_im:
            return new_im


    def newim_callbackfunc(self, image, ctx): #Use different method than numpy one to avoid dupes? E.g get straight from a buffer?
        # new_im = FliSdk_V2.GetRawImageAsNumpyArray(self.cam_context, -1)  # -1 gets most recent image

        ArrayType = ctypes.c_uint16 * self.camdims[0] * self.camdims[1]
        pa = ctypes.cast(image, ctypes.POINTER(ArrayType))
        buffer = np.ndarray((self.camdims[1], self.camdims[0]), dtype=np.uint16, buffer=pa.contents)

        new_im = np.copy(buffer)
        # self.latest_im = new_im ### TODO - double check this can't change im in loggedims_list (another copy?)

        if self.do_logimages and self.nims_lefttolog > 0:
            # self.loggedims_list.append(new_im)
            self.loggedims_cube[self.nims_tolog-self.nims_lefttolog, :, :] = new_im
            # self.loggedims_times.append(time.perf_counter())
            self.loggedims_times_arr[self.nims_tolog-self.nims_lefttolog] = time.perf_counter()
            self.nims_lefttolog -= 1


    # def get_n_images(self, nims, blocking=True, return_ims=False):
    #     self.loggedims_list = []
    #     self.loggedims_times = []
    #     self.nims_lefttolog = 0
    #
    #     # cam0.loggedims_list = []
    #     # cam0.loggedims_times = []
    #
    #     self.do_logimages = True
    #     self.nims_lefttolog = nims
    #
    #     if blocking:
    #         while self.nims_lefttolog > 0:
    #             time.sleep(0.001)
    #         if return_ims:
    #             imcube = np.asarray(self.loggedims_list)
    #             # imcube = np.array(self.loggedims_list, copy=False)
    #             # self.loggedims_list = []
    #             return imcube

    def get_n_images(self, blocking=True, return_ims=False):

        # self.loggedims_times = []
        self.nims_lefttolog = self.nims_tolog

        self.do_logimages = True

        if blocking:
            while self.nims_lefttolog > 0:
                time.sleep(0.001)
            if return_ims:
                # imcube = np.asarray(self.loggedims_list)
                # imcube = np.array(self.loggedims_list, copy=False)
                # self.loggedims_list = []

                # loggedims_cube_copy = np.copy(self.loggedims_cube)
                # self.loggedims_cube = np.zeros((self.nims_tolog, self.camdims[1], self.camdims[0]), dtype=np.uint16)
                # return loggedims_cube_copy

                return self.loggedims_cube


    def set_nims_tolog(self, nims):
        self.nims_tolog = nims
        self.loggedims_cube = np.zeros((self.nims_tolog, self.camdims[1], self.camdims[0]), dtype=np.uint16)
        self.loggedims_times_arr = np.zeros(self.nims_tolog)



# Testing
cam0 = credcam(camera_index=0, verbose=True)


# Get n images
n_ims = 100
cam0.set_nims_tolog(n_ims)
# startime = time.perf_counter()
# imcube = cam0.get_n_images(n_ims, return_ims=True).astype('float32')

# time.sleep(0.5)
starttime = time.perf_counter()
imcube = cam0.get_n_images(return_ims=True).astype('float32')
print('Time to start: %f' % (cam0.loggedims_times_arr[0]-starttime))
print('Total time: %f' % (cam0.loggedims_times_arr[-1]-starttime))

win = (100, 220, 230, 350)
# win = (100, 220, 230, 530)
# win = (100, 220, 230+300, 350+300)
# win = (220, -1, 0, -1)

plt.figure(1)
# immns = np.mean(imcube, axis=(1,2))
# immns = np.std(imcube-imcube[0,:,:], axis=(1,2))
immns = np.std(imcube[:,win[0]:win[1], win[2]:win[3]]-imcube[0,win[0]:win[1], win[2]:win[3]], axis=(1,2))
plt.clf()
plt.plot(immns)

d = np.diff(immns)
print(np.where(d==0))

tdiff = np.diff(cam0.loggedims_times_arr)
plt.figure(2)
plt.clf()
plt.plot(tdiff)


newmn = []
offset = 0
dk = imcube[0,:,:].astype('float32')
plt.figure(3)
plt.pause(0.001)
for k in range(10):#imcube.shape[0]):
    plt.clf()
    plt.imshow(imcube[k+offset,win[0]:win[1], win[2]:win[3]]-dk[win[0]:win[1], win[2]:win[3]], clim=[-80,80])
    plt.title(k+offset)
    # plt.imshow(imcube[k+0, :, :] - dk[:, :])
    # newmn.append(np.std( (imcube[k+150,win[0]:win[1], win[2]:win[3]]-dk[win[0]:win[1], win[2]:win[3]]) ))
    # print(k)
    plt.pause(0.001)




# im = cam0.get_latest_image(return_im=True)
# plt.imshow(cam0.get_latest_image())


# cam0.logimages = False
# d = np.diff(cam0.loggedims_times)
# plt.plot(d[0:1000], '.')






# # Get several ims in a row
# nims = 100
# ims = []
# for k in range(nims):
#     ims.append(cam0.get_latest_image())
# plt.clf()
# plt.imshow(ims[2] - ims[0])



