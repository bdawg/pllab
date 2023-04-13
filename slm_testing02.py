import numpy as np
import time
from plslm import *
import matplotlib
matplotlib.use('TkAgg')


lutfile = 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT'

slm = plslm(lutfile=lutfile)

slm.makeramp(xslope=0.1, showplot=True, sendtoslm=True)

# Images to put on SLM:
# all_images = []
# image1 = np.ones((slm.slmdims[0], slm.slmdims[1])) * 128
# all_images.append(image1)
# image2 = np.random.rand(slm.slmdims[0], slm.slmdims[1]) * 200
# all_images.append(image2)
# all_images.append(image1)
# all_images.append(image2)
# all_images.append(image1)
#
# nloops = 1
# sleeptime = 1
#
# for k in range(nloops):
#     for im in all_images:
#         slm.slmwrite(im, showplot=True)
#         time.sleep(sleeptime)
#
# print('Done')


