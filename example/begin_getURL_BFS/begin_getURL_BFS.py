#!/usr/bin/python
import sys
sys.path.append('Refference')
import streetview_my
import numpy as np
import cv2
import scipy.io as sio
import json
from pprint import pprint


start_panoId = 'OdH7fZyrCUDrpEx4CvLLYA'
panoSet = set()
panoList = []
panoDict = {}
cur = 0

panoList.append(start_panoId)
panoSet.add(start_panoId)
pano = streetview_my.GetPanoramaMetadata(start_panoId)
pano_basic = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'rawDepth':pano.rawDepth}
panoDict['street' + str(cur)] = pano_basic
cur += 1
'''
for link in pano.AnnotationLinks:	
	if (link['PanoId'] in panoSet):
		continue
	panoId = link['PanoId']
	panoList.append(panoId)
	panoSet.add(panoId)
	pano = streetview_my.GetPanoramaMetadata(panoId)	
	pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'rawDepth':pano.rawDepth}
	panoDict['street' + str(cur)] = pano_for_mat
	cur += 1
'''

	
'''
for i in xrange(1,100):
	print i
	pano_for_mat = panoDict['street' + str(i)]

	for link in pano_for_mat['AnnotationLinks']:	
		if (link['PanoId'] in panoSet):
			continue
		panoId = link['PanoId']
		panoList.append(panoId)
		panoSet.add(panoId)
		pano = streetview_my.GetPanoramaMetadata(panoId)	
		pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'rawDepth':pano.rawDepth}	
		panoDict['street' + str(cur)] = pano_for_mat
		cur += 1		
'''
		
		

print panoDict.keys(), len(panoDict.keys())

with open(pano_basic['panoId'] + '.json', 'w') as outfile:
	json.dump(pano_basic, outfile)
	outfile.close()

'''
print pano_for_mat['panoId'];
panoMeta = {}
panoMeta['len'] = len(panoDict.keys())	
panoMeta['data'] = panoDict		
'''

#sio.savemat('streetview_set_onlyMeta.mat', panoMeta)

