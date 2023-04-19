import numpy as np
from ctypes import *
import time

cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")

lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT'


# Basic parameters for calling Create_SDK
bit_depth = c_uint(12)
num_boards_found = c_uint(0)
constructed_okay = c_uint(-1)
is_nematic_type = c_bool(1)
RAM_write_enable = c_bool(1)
use_GPU = c_bool(1)
max_transients = c_uint(20)
board_number = c_uint(1)
wait_For_Trigger = c_uint(0)
flip_immediate = c_uint(0)
timeout_ms = c_uint(5000)

# Call the Create_SDK constructor
# Returns a handle that's passed to subsequent SDK calls
slm_lib.Create_SDK(bit_depth, byref(num_boards_found), byref(constructed_okay), is_nematic_type,
                   RAM_write_enable, use_GPU, max_transients, 0)

if constructed_okay.value == 0:
    print ("Blink SDK did not construct successfully")

if num_boards_found.value == 1:
    print ("Blink SDK was successfully constructed")
    print ("Found %s SLM controller(s)" % num_boards_found.value)

# Set required values
slm_lib.Load_LUT_file(board_number, lutfile.encode('utf-8'))
height = c_uint(slm_lib.Get_image_height(board_number))
width = c_uint(slm_lib.Get_image_width(board_number))
depth = c_uint(slm_lib.Get_image_depth(board_number)) #Bits per pixel
Bytes = c_uint(depth.value//8)
center_x = c_uint(width.value//2)
center_y = c_uint(height.value//2)
imagesize = width.value*height.value*Bytes.value

# Both pulse options can be false, but only one can be true. You either generate a pulse when the new image begins loading to the SLM
# or every 1.184 ms on SLM refresh boundaries, or if both are false no output pulse is generated.
OutputPulseImageFlip = c_uint(0)
OutputPulseImageRefresh = c_uint(0) #only supported on 1920x1152, FW rev 1.8.


# Images to put on SLM:
all_images = []
image1 = np.ones((width.value, height.value)) * 128
all_images.append(image1)
image2 = np.random.rand(width.value, height.value) * 200
all_images.append(image2)
all_images.append(image1)
all_images.append(image2)
all_images.append(image1)

sleeptime = 2

count = 0
for image in all_images:
    print('Displaying image %d' % count)
    slm_image = image.round().astype('uint8').ravel()
    errorval = slm_lib.Write_image(board_number, slm_image.ctypes.data_as(POINTER(c_ubyte)),
                                 imagesize, wait_For_Trigger, flip_immediate,
                                 OutputPulseImageFlip, OutputPulseImageRefresh, timeout_ms)
    if (errorval == -1):
        print("DMA Failed")

    # check the buffer is ready to receive the next image
    errorval = slm_lib.ImageWriteComplete(board_number, timeout_ms)
    if (errorval == -1):
        print("ImageWriteComplete failed, trigger never received?")

    print('Image write complete, pausing until next image')
    count += 1
    time.sleep(sleeptime)

slm_lib.Delete_SDK()
print('Done.')
