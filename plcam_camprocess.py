import numpy as np
import time
from multiprocessing import shared_memory
from plcams import credcam
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()


class plcam_camprocess():
    def __init__(self, camid, cam_commsl_shmname, cam_imshm_shmname, cam_settings=None, verbose=False,
                 delays=(12,3), trigger_mode='external', poll_time=0.01):
        self.camid = camid
        cam_syncdelay_ms = delays[0]
        extra_delay_ms = delays[1]
        self.wait_time_ms = cam_syncdelay_ms + extra_delay_ms
        self.poll_time = poll_time
        self.runmode = True

        # Set up shared memory
        self.cam_commsl = shared_memory.ShareableList(name=cam_commsl_shmname)
        self.cam_imshm_obj = shared_memory.SharedMemory(name=cam_imshm_shmname)
        self.cube_nims = self.cam_commsl[2]
        self.camdims = [self.cam_commsl[3], self.cam_commsl[4]]
        # self.cam_imshm = np.ndarray((self.cube_nims, self.camdims[1], self.camdims[0]), dtype=np.int16,
        #                             buffer=cam_imshm_obj.buf)
        if self.cam_commsl[11] is not None:
            self.cropdims = [self.cam_commsl[11], self.cam_commsl[12], self.cam_commsl[13],self.cam_commsl[14]]
        else:
            self.cropdims = None

        # Set up camera
        self.cam = credcam(camera_id=camid, verbose=verbose, cam_settings=cam_settings, cropdims=self.cropdims,
                           buffersize_ims=self.cube_nims)
        if trigger_mode == 'external':
            self.cam.external_trigger(enabled=True, syncdelay=cam_syncdelay_ms, verbose=verbose)
            self.cam.reset_buffer()
        else:
            print('camprocess ' + self.camid + ': Error: only external trigger is supported so far for SHM mode...')
            return

        # ##### DEBUG #####
        # print(' ')
        # print('Debug from plcam_camprocess')
        # print('Camera ' + camid)
        # print([self.cam_commsl[2], self.cam_commsl[3], self.cam_commsl[4],
        #        self.cam_commsl[5], self.cam_commsl[10]])
        # #####

        if verbose:
            print('camprocess ' + self.camid + ': setup complete')
        self.cam_commsl[5] = 1

        self.main_loop(verbose=verbose)


    def main_loop(self, verbose=False):
        if verbose:
            print('camprocess ' + self.camid + ': waiting for command')
            # print('Try sharing testcube...')
            # testcube = np.ones((self.cube_nims, self.camdims[1], self.camdims[0]), dtype=np.int16)
            # self.cam_imshm[:] = testcube
            # print('Testcube copied.')

        mainloop_count = 0
        while self.runmode:
            # Check for a camera command
            if self.cam_commsl[0] is not None:
                resp = self.cam.send_command(self.cam_commsl[0], return_response=True, verbose=False)
                if verbose:
                    print(' ')
                    print('camprocess ' + self.camid + ': sent command: ' + self.cam_commsl[0])
                    print('camprocess ' + self.camid + ': response: ' + resp)
                self.cam_commsl[0] = None
                self.cam_commsl[1] = resp

            # Check for image cube acquire command
            if self.cam_commsl[10] == 1:
                self.cam_commsl[5] = 0
                if self.cam_commsl[6] is not None:
                    cur_cube_nims = self.cam_commsl[6]
                else:
                    cur_cube_nims = self.cube_nims

                # starttime = time.time()
                self.cam.reset_buffer()
                print('cur_cube_nims: ')
                print(cur_cube_nims)
                while self.cam.check_nims_buffer() < cur_cube_nims:
                    if verbose:
                        print('camprocess ' + self.camid + ': Acquired image %d of %d\n' %
                          (self.cam.check_nims_buffer(), cur_cube_nims))
                    if cur_cube_nims > 1:
                        time.sleep(0.5)
                    # TODO - Implement timeout
                cam_imshm = np.ndarray((self.cube_nims, self.camdims[1], self.camdims[0]), dtype=np.int16,
                                    buffer=self.cam_imshm_obj.buf)
                if cur_cube_nims == self.cube_nims:
                    cam_imshm[:] = np.copy(self.cam.get_buffer_images()[:cur_cube_nims,:,:])
                else:
                    cam_imshm[:] = np.zeros((self.cube_nims, self.camdims[1], self.camdims[0]), dtype=np.int16)
                    cam_imshm[:cur_cube_nims,:,:] = np.copy(self.cam.get_buffer_images()[:cur_cube_nims, :, :])

                print('Returned from get_buffer')
                self.cam_commsl[10] = 0
                if verbose:
                    print('camprocess ' + self.camid + ': Acquisition complete')
                self.cam_commsl[5] = 1
                self.cam_commsl[6] = None

            if verbose:
                if mainloop_count % 10000 == 0:
                    print('camprocess ' + self.camid + ': still alive')

            mainloop_count += 1
            time.sleep(self.poll_time)



if __name__ == '__main__':
    args = sys.argv
    camid = args[1]
    cam_commsl_shmname = args[2]
    cam_imshm_shmname = args[3]
    if args[4] == '1':
        verbose = True
    else:
        verbose = False

    plcam = plcam_camprocess(camid, cam_commsl_shmname, cam_imshm_shmname, verbose=verbose)












