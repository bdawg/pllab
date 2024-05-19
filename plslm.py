import numpy as np
import time
from ctypes import *
import matplotlib
from scipy.signal import square
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()


class plslm:
    def __init__(self, lutfile=None, slmtimeout=5000, slmoffset=0, testmode=False, slmloc=None):
        if lutfile is None:
            lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6658_at1550_75C.LUT'

        if not testmode:
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
            self.slmdims = (int(self.width.value), int(self.height.value))
            # center_x = c_uint(self.width.value // 2)
            # center_y = c_uint(self.height.value // 2)
            # OutputPulseImageFlip = c_uint(0)
            self.nextim = None
            self.slmoffset = slmoffset
        else:
            self.slmdims = (1024,1024)
            self.slmoffset = 0

        if slmloc is not None:
            # slmloc = np.array([slm_centre[0], slm_centre[1], slm_rad])
            print('Note: using SLM SUBREGION radius %d, centre(%d,%d).' % (slmloc[2], slmloc[0], slmloc[1]))
        self.slmloc = slmloc


    def close(self):
        self.slmobj.Delete_SDK()
        print('SLM closed.')


    def load_lut(self, lutfile):
        self.slmobj.Load_LUT_file(self.board_number, lutfile.encode('utf-8'))


    def rad2im(self, im_rad, slm_range=4*np.pi, centre_zero=True, set_nextim=False, wrap=True):
        if wrap:
            if np.min(im_rad) < -slm_range/2 or np.max(im_rad) > slm_range/2:
                print('Warning: image outside SLM phase range will be wrapped')
                im_rad = np.mod(im_rad + slm_range/2, slm_range) - slm_range/2
        im = im_rad / slm_range * 255
        if centre_zero:
            im += 127.5
        if np.min(im) < 0 or np.max(im) > 255:
            print('Warning: SLM image values outside 0-255 range, will be clipped')
            im[im < 0] = 0
            im[im > 255] = 255
        im = np.round(im)
        im = im.astype('uint8')
        if set_nextim:
            self.nextim = im
        return im


    def slmwrite(self, im=None, showplot=False, skip_readycheck=False, fignum=1):
        if im is None:
            im = self.nextim
        else:
            self.nextim = im
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
            plt.figure(fignum)
            plt.clf()
            imtoshow = slm_image.reshape(self.slmdims[0], self.slmdims[1])
            cmap = 'twilight_shifted'
            # cmap = 'viridis'
            plt.imshow(imtoshow, interpolation='None', cmap=cmap, clim=[0,255])
            plt.colorbar()
            plt.pause(0.001)


    def makeramp(self, xslope=1, yslope=0, dir=0, showplot=False, sendtoslm=False, return_im=False):
        if self.slmloc is not None:
            print('WARNING - SLM subregion not yet suported!')
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
        if return_im:
            return im


    def makeramp_rad(self, slope_rmsrad, angle_deg, centre_val=0, showplot=False, showslmplot=False,
                     sendtoslm=False, return_im=False, fignum=2):
        if self.slmloc is not None:
            dims = [self.slmloc[2]*2, self.slmloc[2]*2]
        else:
            dims = self.slmdims
        xslope = np.cos(angle_deg / 180 * np.pi)
        yslope = np.sin(angle_deg / 180 * np.pi)
        Y, X = np.mgrid[:dims[0], :dims[1]]
        im = xslope * X + yslope * Y
        im = im / np.std(im) * slope_rmsrad
        im = im - np.mean(im)
        im = im + centre_val
        if self.slmloc is not None:
            slm_centre = self.slmloc[:2]
            slm_rad = self.slmloc[2]
            pim = np.zeros((self.slmdims[0], self.slmdims[1]))
            pim[slm_centre[0] - slm_rad:slm_centre[0] + slm_rad, \
                slm_centre[1] - slm_rad:slm_centre[1] + slm_rad] = im
            im = pim
        self.nextim = self.rad2im(im)
        if sendtoslm:
            self.slmwrite(showplot=showslmplot)
        if showplot:
            plt.figure(fignum)
            plt.clf()
            plt.imshow(im)
            plt.colorbar()
            plt.pause(0.001)
        if return_im:
            return im


    def make_intensity_checkerboard(self, intensity_val, checkerboard_cellsz=8, offset_rad=0.01,
                                    apply_sinecorr=True, return_rad=True):
        ncells = int(self.slmdims[0] / (checkerboard_cellsz * 2))
        checkerboard = np.kron([[1, 0] * ncells, [0, 1] * ncells] * ncells,
                               np.ones((checkerboard_cellsz, checkerboard_cellsz))) * np.pi - np.pi / 2
        # checkerboard = np.kron([[1, 0] * ncells, [0, 1] * ncells] * ncells,
        #                        np.ones((checkerboard_cellsz, checkerboard_cellsz))) * np.pi - np.pi / 2

        if apply_sinecorr:
            checkerboard_coeff = (np.arcsin(2*intensity_val-1) + np.pi/2) / np.pi
        else:
            checkerboard_coeff = intensity_val
        chkbd_cur_rad = checkerboard * (1 - checkerboard_coeff) + offset_rad
        # chkbd_cur_rad = checkerboard * checkerboard_coeff + offset_rad
        # if centre_zero:
        #     chkbd_cur_rad = chkbd_cur_rad - np.mean(chkbd_cur_rad)
        chkbd_cur = self.rad2im(chkbd_cur_rad)
        if return_rad:
            return chkbd_cur_rad
        else:
            return chkbd_cur


    def make_incoh_psfs(self, seps, angles, intensities, ld2r=None):
        if ld2r is not None:
            self.ld2r = ld2r
        else:
            ld2r = self.ld2r

        num_subims = len(seps)
        slm_subims = np.zeros((num_subims, self.slmdims[0], self.slmdims[1]))
        # slm_subims_rad = np.zeros((num_subims, self.slmdims[0], self.slmdims[1]))
        for k in range(num_subims):
            slope = seps[k] * ld2r
            spot_rad = self.makeramp_rad(slope, angles[k], return_im=True)
            grid_rad = self.make_intensity_checkerboard(intensities[k])
            im_rad = spot_rad + grid_rad
            # slm_subims_rad[k, :, :] = im_rad
            slm_subims[k, :, :] = self.rad2im(im_rad)

        return slm_subims


    def makestripes(self, period=10, angle=0, ampl=100, phi=0, offset=0, type='square', showplot=False, sendtoslm=False,
                    return_im=False):
        if self.slmloc is not None:
            print('WARNING - SLM subregion not yet suported!')
        # Ampl is the peak-to-valley amplitude

        # phi = np.pi/2
        x = np.arange(-self.slmdims[0]/2, self.slmdims[0]/2)
        x_rad = x * (1 / period) * 2 * np.pi

        ## Old behaviour - centred at 127
        # if type == 'square':
        #     y = square(x_rad + phi) * ampl/2 + 127
        # elif type == 'sine':
        #     y = np.sin(x_rad + phi) * ampl/2 + 127
        # else:
        #     print('Unknown type specified')
        #     return

        if type == 'square':
            y = square(x_rad + phi) * ampl/2 + ampl/2 + offset
        elif type == 'sine':
            y = np.sin(x_rad + phi) * ampl/2 + ampl/2 + offset
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

























