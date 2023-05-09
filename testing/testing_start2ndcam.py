
import matplotlib.pyplot as plt
import numpy as np
import time
from plslm import plslm
from plcams import credcam
import matplotlib
matplotlib.use('TkAgg')
import FliSdk_V2

cam_index1 = 1
cam_context1 = FliSdk_V2.Init()
listOfGrabbers = FliSdk_V2.DetectGrabbers(cam_context1)
listOfCameras = FliSdk_V2.DetectCameras(cam_context1)
print('Using camera ' + listOfCameras[cam_index1])
num_cameras = len(listOfCameras)
errorval = FliSdk_V2.SetCamera(cam_context1, listOfCameras[cam_index1])
FliSdk_V2.SetMode(cam_context1, FliSdk_V2.Mode.Full) # Enables grab and config
errorval = FliSdk_V2.Update(cam_context1)
FliSdk_V2.Start(cam_context1)