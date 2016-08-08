#!/usr/bin/python2
import numpy as np
import json
import sys
sys.path.append('module')	# use the module under 'module'
import google_store

with open('src/json/dashcam/namelist_50.json') as data_file:    
    namelist_50 = json.load(data_file)
    for index, namelist in enumerate(namelist_50):
        if index == 5:
            [fileIndex, fileID] = namelist.split(',')
    data_file.close()

with open('src/json/dashcam/deep_match/' + fileID + '/info_3d.json') as data_file:    
    info_3d = json.load(data_file)
    data_file.close()


pathPoint_set_info3d = set()
for img, latlon in info_3d.items():
		for latlon_element in latlon.keys():		
			if latlon_element not in pathPoint_set_info3d:	
				pathPoint_set_info3d.add(latlon_element)
		
print index, fileIndex, fileID
print len(pathPoint_set_info3d), pathPoint_set_info3d

google_store.info_3d(fileID, pathPoint_set_info3d)
