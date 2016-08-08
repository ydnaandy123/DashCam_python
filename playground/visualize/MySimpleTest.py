#!/usr/bin/python

import streetview_my
import numpy as np
import cv2
import scipy.io as sio
import sys
'''
a = np.zeros((200,500));
b = np.zeros((200,500));
s = 'hello'
cv2.imwrite('a.jpg',a);
sio.savemat('a.mat', {'a':a, 'b':b})
sio.savemat('s.mat', {'s':s, 'abc':a})
'''
pano = streetview_my.GetPanoramaMetadata("L5QY_nh7wOUz3Fd4GhpLaA")

e = np.random.random((pano.DepthHeader['numPlanes'],3)) * 255
height = pano.DepthHeader['panoHeight']
width = pano.DepthHeader['panoWidth']
depthIndex = pano.DepthMapIndices
depthPlane = pano.DepthMapPlanes
data = np.zeros((height, width, 3))
for h in range(0, height):
    for w in range(0, width):
        dIndex = depthIndex[h*width +w]
        data[h][w][:] = e[dIndex, :]

'''
data = np.zeros((height, width, 3))
print (height, width)
for w in range(0, width):
    for h in range(0, height):
        try:
            dIndex = depthIndex[h +w*height]*20
        except:
            print (len(depthIndex))
            print (h,w,w*width)
            sys.exit()
        data[h][w][:] = [dIndex,dIndex,dIndex]         
'''
cv2.imwrite('depth.png',data)
