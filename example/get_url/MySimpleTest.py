#!/usr/bin/python

import streetview_my
import numpy as np
import cv2
import scipy.io as sio


'''
pano = streetview_my.GetPanoramaMetadata("OdH7fZyrCUDrpEx4CvLLYA")
sio.savemat('streetview.mat', {'rawDepth':pano.rawDepth, 'AnnotationLinks':pano.AnnotationLinks, 'DepthMapIndices':pano.DepthMapIndices, 'DepthMapPlanes':pano.DepthMapPlanes, 'DepthHeader':pano.DepthHeader})
'''
'''
x.astype(int)
array([1, 2, 2])
'''
'''
a = np.zeros((200,500));
b = np.zeros((200,500));
s = 'hello'
cv2.imwrite('a.jpg',a);
sio.savemat('a.mat', {'a':a, 'b':b})
sio.savemat('s.mat', {'s':s, 'abc':a})
'''

'''
pano = streetview_my.GetPanoramaMetadata("OdH7fZyrCUDrpEx4CvLLYA")
import json
with open('data.txt', 'w') as outfile:
    json.dump(pano.raw, outfile)
'''
