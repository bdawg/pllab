import numpy as np
import time
from ctypes import *
import matplotlib
from scipy.signal import square
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()


class plslm:
    def __init__(self, lutfile=None, slmtimeout=5000, slmoffset=127):
        if lutfile is None:
            lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'

        cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
        self.slmobj = CDLL("Blink_C_wrapper")

        # Basic parameters for calling Create_SDK
        bit_depth = c_uint(12)
        num_boards_found = c_uint(0)
        constructed_okay = c_uint(-1)
        is_nematic_type = c_bool(1)
        RAM_write_enable = c_bool(1)
        use_GPU = c_bool(1)
        max_transients = c_uint(20)
        self.board_number = c_uint(1)
        self.wait_For_Trigger = c_uint(0)
        self.flip_immediate = c_uint(0)
        self.OutputPulseImageFlip = c_uint(1)
        self.OutputPulseImageRefresh = c_uint(0)  # only supported on 1920x1152, FW rev 1.8.
        self.timeout_ms = c_uint(slmtimeout)

        # Call the Create_SDK constructor
        # Returns a handle that's passed to subsequent SDK calls
        self.slmobj.Create_SDK(bit_depth, byref(num_boards_found), byref(constructed_okay), is_nematic_type,
                           RAM_write_enable, use_GPU, max_transients, 0)

        if num_boards_found.value == 1:
            print("SLM initialisation successful")
        else:
            print('Error initialising SLM')

        # Set required values
        self.slmobj.Load_LUT_file(self.board_number, lutfile.encode('utf-8'))
        self.height = c_uint(self.slmobj.Get_image_height(self.board_number))
        self.width = c_uint(self.slmobj.Get_image_width(self.board_number))
        depth = c_uint(self.slmobj.Get_image_depth(self.board_number))  # Bits per pixel
        Bytes = c_uint(depth.value // 8)
        self.imagesize = self.width.value * self.height.value * Bytes.value
        self.slmdims = (self.width.value, self.height.value)
        # center_x = c_uint(self.width.value // 2)
        # center_y = c_uint(self.height.value // 2)
        # OutputPulseImageFlip = c_uint(0)
        self.nextim = None
        self.slmoffset = slmoffset


    def close(self):
        self.slmobj.Delete_SDK()
        print('SLM closed.')


    def slmwrite(self, im=None, showplot=False, skip_readycheck=False):
        if im is None:
            im = self.nextim
        im = im + self.slmoffset
        slm_image = im.round().astype('uint8').ravel()
        errorval = self.slmobj.Write_image(self.board_number, slm_image.ctypes.data_as(POINTER(c_ubyte)),
                                       self.imagesize, self.wait_For_Trigger, self.flip_immediate,
                                       self.OutputPulseImageFlip, self.OutputPulseImageRefresh, self.timeout_ms)
        if (errorval == -1):
            print("SLM write failed")

        if not skip_readycheck:
            # check the buffer is ready to receive the next image
            errorval = self.slmobj.ImageWriteComplete(self.board_number, self.timeout_ms)
            if (errorval == -1):
                print("ImageWriteComplete failed, trigger never received?")

        if showplot:
            plt.clf()
            imtoshow = slm_image.reshape(self.slmdims[0], self.slmdims[1])
            plt.imshow(imtoshow, interpolation='None', cmap='twilight_shifted', clim=[0,255])
            plt.colorbar()
            plt.pause(0.001)


    def makeramp(self, xslope=1, yslope=0, dir=0, showplot=False, sendtoslm=False):
        Y, X = np.mgrid[:self.slmdims[0], :self.slmdims[1]]
        im = xslope * X + yslope * Y
        # im -= im[self.slmdims[0] // 2, self.slmdims[1] // 2]
        self.nextim = im
        if sendtoslm:
            self.slmwrite(showplot=showplot)
        elif showplot:
            plt.clf()
            plt.imshow(im)
            plt.colorbar()
            plt.pause(0.001)


    def makestripes(self, period=10, angle=0, ampl=100, phi=0, type='square', showplot=False, sendtoslm=False,
                    return_im=False):
        # phi = np.pi/2
        x = np.arange(-self.slmdims[0]/2, self.slmdims[0]/2)
        x_rad = x * (1 / period) * 2 * np.pi

        if type == 'square':
            y = square(x_rad + phi) * ampl
        elif type == 'sine':
            y = np.sin(x_rad + phi) * ampl
        else:
            print('Unknown type specified')
            return
        im = np.broadcast_to(y, (self.slmdims[0], self.slmdims[1]))

        self.nextim = im
        if sendtoslm:
            self.slmwrite(showplot=showplot)
        elif showplot:
            plt.clf()
            plt.imshow(im)
            plt.colorbar()
            plt.pause(0.001)

        if return_im:
            return im

























