#!/usr/bin/python
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')
import streetview_my
import numpy as np
import cv2
import scipy.io as sio
import json

fileID = '000731'
panoSet = set()
panoList = []
panoDict = {}
cur = 0

with open('example/url_require/' + fileID + '_lat_lon_result.json') as data_file:    
    SFM_path = json.load(data_file)

pathPoint_set = []
for key, value in SFM_path.iteritems():
    	pathPoint_set.append(value)
print len(pathPoint_set)

for i in xrange(0,len(pathPoint_set)-1):
	pathPoint = pathPoint_set[i]
	pano = streetview_my.GetPanoramaMetadata(lat = pathPoint[0], lon = pathPoint[1])
	if pano.PanoId not in panoSet:
		panoSet.add(pano.PanoId)
		pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg,\
						'rawDepth':pano.rawDepth, 'DepthMapIndices':pano.DepthMapIndices, 'DepthMapPlanes':pano.DepthMapPlanes, 'DepthHeader':pano.DepthHeader}	
		#pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg}
		panoDict['street' + str(cur)] = pano_for_mat
		cur = cur+1

print panoDict.keys(), len(panoDict.keys())

panoMeta = {}
panoMeta['len'] = len(panoDict.keys())	
panoMeta['data'] = panoDict	

sio.savemat('streetview_set_' + fileID + '.mat', panoMeta)	

# PahtPoint
pathPoint = []
for key, value in panoDict.iteritems():
    	pathPoint.append(value['Lat']+','+ value['Lon'])
print pathPoint
with open('pathPoint_' + fileID + '.json', 'w') as outfile:
    json.dump(pathPoint, outfile)	

'''
panoList.append(start_panoId)
panoSet.add(start_panoId)
pano = streetview_my.GetPanoramaMetadata(start_panoId)
pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg,'rawDepth':pano.rawDepth, 'DepthMapIndices':pano.DepthMapIndices, 'DepthMapPlanes':pano.DepthMapPlanes, 'DepthHeader':pano.DepthHeader}	
#pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg}
panoDict['street' + str(cur)] = pano_for_mat
cur += 1

for link in pano.AnnotationLinks:	
	if (link['PanoId'] in panoSet):
		continue
	panoId = link['PanoId']
	panoList.append(panoId)
	panoSet.add(panoId)
	pano = streetview_my.GetPanoramaMetadata(panoId)
	pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg, 'rawDepth':pano.rawDepth, 'DepthMapIndices':pano.DepthMapIndices, 'DepthMapPlanes':pano.DepthMapPlanes, 'DepthHeader':pano.DepthHeader}	
	#pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg}
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
		pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg, 'rawDepth':pano.rawDepth, 'DepthMapIndices':pano.DepthMapIndices, 'DepthMapPlanes':pano.DepthMapPlanes, 'DepthHeader':pano.DepthHeader}	
		#pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg}	
		panoDict['street' + str(cur)] = pano_for_mat
		cur += 1		

		
		

print panoDict.keys(), len(panoDict.keys())

panoMeta = {}
panoMeta['len'] = len(panoDict.keys())	
panoMeta['data'] = panoDict	


sio.savemat('streetview_set_' + fileID + '.mat', panoMeta)

# PahtPoint
pathPoint = []
for key, value in panoDict.iteritems():
    	pathPoint.append(value['Lat']+','+ value['Lon'])
print pathPoint
with open('pathPoint_' + fileID + '.json', 'w') as outfile:
    json.dump(pathPoint, outfile)


def pano_require(cur):
    	print cur, len(pathPoint_set)
	pathPoint = pathPoint_set[cur]
	pano = streetview_my.GetPanoramaMetadata(lat = pathPoint[0], lon = pathPoint[1])
	if pano.PanoId not in panoSet:
		panoSet.add(pano.PanoId)
		pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg,\
						'rawDepth':pano.rawDepth, 'DepthMapIndices':pano.DepthMapIndices, 'DepthMapPlanes':pano.DepthMapPlanes, 'DepthHeader':pano.DepthHeader}	
		#pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg}
		panoDict['street' + str(len(panoSet)-1)] = pano_for_mat
	cur = cur+1
	if cur < len(pathPoint_set):
		pano_require(cur)

pano_require(0)
'''