import sys
sys.path.append('Refference')
import streetview_my

import json
import scipy.io as sio
from pprint import pprint
import numpy as np
import cv2

import os
panoPath = os.getcwd() + '/src/panoData/'
onlyfiles = [f for f in os.listdir(panoPath) if os.path.isfile(os.path.join(panoPath, f))]

for fileName in onlyfiles:
    print panoPath + fileName
    with open('OdH7fZyrCUDrpEx4CvLLYA.json') as data_file:
        data = json.load(data_file)
        pprint(data)
        data_file.close();
