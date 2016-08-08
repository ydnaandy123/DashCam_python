#!/usr/bin/python

import streetview_my
import numpy as np
import cv2
import scipy.io as sio
import json

fileID = '000731'
start_panoId = 'y5KKray3BgfMR8jZohGMGA'
requireNum = 30
panoSet = set()
panoList = []
panoDict = {}
cur = 0

panoList.append(start_panoId)
panoSet.add(start_panoId)
pano = streetview_my.GetPanoramaMetadata(start_panoId)
#pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'rawDepth':pano.rawDepth, 'AnnotationLinks':pano.AnnotationLinks, 'DepthMapIndices':pano.DepthMapIndices, 'DepthMapPlanes':pano.DepthMapPlanes, 'DepthHeader':pano.DepthHeader}	
pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks}
panoDict['street' + str(cur)] = pano_for_mat
cur += 1

for link in pano.AnnotationLinks:	
	if (link['PanoId'] in panoSet):
		continue
	panoId = link['PanoId']
	panoList.append(panoId)
	panoSet.add(panoId)
	pano = streetview_my.GetPanoramaMetadata(panoId)
	#pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'rawDepth':pano.rawDepth, 'AnnotationLinks':pano.AnnotationLinks, 'DepthMapIndices':pano.DepthMapIndices, 'DepthMapPlanes':pano.DepthMapPlanes, 'DepthHeader':pano.DepthHeader}	
	pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks}
	panoDict['street' + str(cur)] = pano_for_mat
	cur += 1

	

for i in xrange(1,requireNum):
	print i
	pano_for_mat = panoDict['street' + str(i)]

	for link in pano_for_mat['AnnotationLinks']:	
		if (link['PanoId'] in panoSet):
			continue
		panoId = link['PanoId']
		panoList.append(panoId)
		panoSet.add(panoId)
		pano = streetview_my.GetPanoramaMetadata(panoId)
		#pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'rawDepth':pano.rawDepth, 'AnnotationLinks':pano.AnnotationLinks, 'DepthMapIndices':pano.DepthMapIndices, 'DepthMapPlanes':pano.DepthMapPlanes, 'DepthHeader':pano.DepthHeader}	
		pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks}	
		panoDict['street' + str(cur)] = pano_for_mat
		cur += 1		

		
		

print panoDict.keys(), len(panoDict.keys())

panoMeta = {}
panoMeta['len'] = len(panoDict.keys())	
panoMeta['data'] = panoDict		

sio.savemat('streetview_set_onlyMeta_' + fileID + '.mat', panoMeta)

