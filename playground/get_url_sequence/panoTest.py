#!/usr/bin/python

import streetview_my
import numpy as np
import cv2
import scipy.io as sio
import json


start_panoId = 'OdH7fZyrCUDrpEx4CvLLYA'
pano = streetview_my.GetPanoramaMetadata(start_panoId)
pano_tile = streetview_my.GetPanoramaTile(pano.PanoId, 2, 0, 0)
with open("a.jpg", 'w') as f:
	f.write(pano_tile)
	'''
cv2.imwrite('a.jpg',pano_tile);
'''
