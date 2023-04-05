fli_sdk_path = 'C:\\Program Files\\FirstLightImaging\\FliSdk\\Python\\lib'
sys.path.append(fli_sdk_path)

import FliSdk_V2

context = FliSdk_V2.Init()

print("Detection of grabbers...")
listOfGrabbers = FliSdk_V2.DetectGrabbers(context)

if len(listOfGrabbers) == 0:
    print("No grabber detected, exit.")
    exit()

print("Done.")
print("List of detected grabber(s):")

for s in listOfGrabbers:
    print("- " + s)

print("Detection of cameras...")
listOfCameras = FliSdk_V2.DetectCameras(context)

if len(listOfCameras) == 0:
    print("No camera detected, exit.")
    exit()

print("Done.")
print("List of detected camera(s):")

i = 0
for s in listOfCameras:
    print("- " + str(i) + " -> " + s)
    i = i + 1

cameraIndex = int(input("Which camera to use? (0, 1, ...) "))
print("Setting camera: " + listOfCameras[cameraIndex])
ok = FliSdk_V2.SetCamera(context, listOfCameras[cameraIndex])

if not ok:
    print("Error while setting camera.")
    exit()

print("Setting mode full.")
FliSdk_V2.SetMode(context, FliSdk_V2.Mode.Full)

print("Updating...")
ok = FliSdk_V2.Update(context)

if not ok:
    print("Error while updating SDK.")
    exit()

print("Done.")

fps = 0

if FliSdk_V2.IsSerialCamera(context):
    res, fps = FliSdk_V2.FliSerialCamera.GetFps(context)
elif FliSdk_V2.IsCblueSfnc(context):
    res, fps = FliSdk_V2.FliCblueSfnc.GetAcquisitionFrameRate(context)
print("Current camera FPS: " + str(fps))

val = input("FPS to set? ")
if val.isnumeric():
    if FliSdk_V2.IsSerialCamera(context):
        FliSdk_V2.FliSerialCamera.SetFps(context, float(val))
    elif FliSdk_V2.IsCblueSfnc(context):
        FliSdk_V2.FliCblueSfnc.SetAcquisitionFrameRate(context, float(val))

if FliSdk_V2.IsSerialCamera(context):
    res, fps = FliSdk_V2.FliSerialCamera.GetFps(context)
elif FliSdk_V2.IsCblueSfnc(context):
    res, fps = FliSdk_V2.FliCblueSfnc.GetAcquisitionFrameRate(context)
print("FPS read: " + str(fps))

if FliSdk_V2.IsCredTwo(context) or FliSdk_V2.IsCredThree(context):
    res, response = FliSdk_V2.FliSerialCamera.SendCommand(context, "mintint raw")
    minTint = float(response)

    res, response = FliSdk_V2.FliSerialCamera.SendCommand(context, "maxtint raw")
    maxTint = float(response)

    res, response = FliSdk_V2.FliSerialCamera.SendCommand(context, "tint raw")

    print("Current camera tint: " + str(float(response)*1000) + "ms")

    val = input("Tint to set? (between " + str(minTint*1000) + "ms and " + str(maxTint*1000)+ "ms) ")
    if val.isnumeric():
        res, response = FliSdk_V2.FliSerialCamera.SendCommand(context, "set tint " + str(float(val)/1000))

    res, response = FliSdk_V2.FliSerialCamera.SendCommand(context, "tint raw")
    print("Current camera tint: " + str(float(response)*1000) + "ms")
elif FliSdk_V2.IsCblueSfnc(context):
    res, tint = FliSdk_V2.FliCblueSfnc.GetExposureTime(context)
    print("Current camera tint: " + str(tint/1000) + "ms")

val = input("How much images to read? ")
if not val.isnumeric():
    val = 600

FliSdk_V2.ImageProcessing.EnableAutoClip(context, -1, True)
FliSdk_V2.ImageProcessing.SetColorMap(context, -1, "RAINBOW")
FliSdk_V2.Start(context)

# for i in range(int(val)):
#     image = FliSdk_V2.GetProcessedImage(context, -1) #-1 to get the last image in the buffer
#     FliSdk_V2.Display8bImage(context, image, "image 8b")
#     image = FliSdk_V2.GetRawImage(context, -1)
#     FliSdk_V2.Display16bImage(context, image, "image 16b", False)

import time
npim1 = FliSdk_V2.GetRawImageAsNumpyArray(context, -1)
time.sleep(0.01)
npim2 = FliSdk_V2.GetRawImageAsNumpyArray(context, -1)

FliSdk_V2.Stop(context)
FliSdk_V2.Exit(context)

import matplotlib
# %matplotlib tk
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
plt.imshow(npim1)