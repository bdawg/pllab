
import numpy as np
import sys
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import FliSdk_V2

# Camera to use:
cam_index = 1

# A command to send
cam_cmd = 'set tint 1'

cam_context = FliSdk_V2.Init()
listOfGrabbers = FliSdk_V2.DetectGrabbers(cam_context)
listOfCameras = FliSdk_V2.DetectCameras(cam_context)
num_cameras = len(listOfCameras)

print('%d cameras detected: ' % num_cameras)
for k in range(num_cameras):
    print('%d: '%k + listOfCameras[k])
print('Using camera %d' % cam_index)

errorval = FliSdk_V2.SetCamera(cam_context, listOfCameras[cam_index])
FliSdk_V2.SetMode(cam_context, FliSdk_V2.Mode.Full) # Enables grab and config
errorval = FliSdk_V2.Update(cam_context)

# Get a value via API
errorval, fps = FliSdk_V2.FliSerialCamera.GetFps(cam_context)
print('Fps: %f' % fps)

# # Set a value via API
# new_fps = 600
# FliSdk_V2.FliSerialCamera.SetFps(cam_context, float(new_fps))
# errorval, fps = FliSdk_V2.FliSerialCamera.GetFps(cam_context)
# print('Fps: %f' % fps)
#
# # Send and receive using serial commands
# errorval, response = FliSdk_V2.FliSerialCamera.SendCommand(cam_context, cam_cmd)
# print('Response: ' + response)
#
# # print('Starting cam in 3 seconds')
# # time.sleep(3)
# FliSdk_V2.Start(cam_context)
# time.sleep(0.01) # Give it a chance to grab some frames
#
# # Get an image
# new_im = FliSdk_V2.GetRawImageAsNumpyArray(cam_context, -1) # -1 gets most recent image
#
# plt.imshow(new_im)
# plt.pause(0.001)
# # print('Got image, pausing for 3 seconds')
# # time.sleep(3)
#
# FliSdk_V2.Stop(cam_context)
# FliSdk_V2.Exit(cam_context)
#
#
#
