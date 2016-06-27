import sys
sys.path.append('Refference')
import streetview_my

import json
import scipy.io as sio
from pprint import pprint
import numpy as np
import cv2

#DepthMapPlanes

pano = streetview_my.GetPanoramaMetadata(lat=27.683528, lon=-99.580078)
#print pano.PanoId
#print len(pano.DepthMapPlanes)
#print len(pano.DepthMapIndices)
print 256*512

height = 256; width = 512
indice_num = len(pano.DepthMapPlanes);
depthMap = np.zeros((height,width));
for y in range(height):
    for x in range(width):
        planeIdx = pano.DepthMapIndices[y*width + x];
        depthMap[y][x] = planeIdx*15
cv2.imwrite('depthMap.jpg',depthMap);

# Open a file
#with open('data.txt', 'w') as outfile:
#    json.dump(pano.DepthMapPlanes, outfile)

# Close opend file
#outfile.close()