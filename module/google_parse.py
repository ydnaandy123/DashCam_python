#!/usr/bin/python3
import numpy as np
import numpy.matlib
import cv2
import zlib
import base64
import struct
import os
import json
#
import base_process
class StreetView3DRegion:
	def __init__(self, fileID):    		
		fname = '/home/andy/src/Google/panometa/' + fileID + '/fileMeta.json'
		if os.path.isfile(fname) :
			print ('Successfully find the existing region"' + fileID + '"(accroding to the fileMeta):')
			with open(fname) as data_file:    
				fileMeta = json.load(data_file)
				self.panoDict = fileMeta['id2GPS']
				data_file.close() 		
		else:
			print ('Fail to open the file or path doesn\'t exit')
	def createTopoloy(self):
		id2GPS = self.panoDict	
		panoNum = len(id2GPS)	
		data = np.zeros((panoNum), dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])
		ECEF = []
		for id in id2GPS:
			GPS = id2GPS[id]
			ECEF.append(base_process.geo2ECEF(float(GPS[0]), float(GPS[1])))
		data['a_color'] = [1,1,1]
		data['a_position'] = np.asarray(ECEF, dtype = np.float32)
		#data['a_position'] -= data[0]['a_position']
		self.topology = data
		return data
class StreetView3D:   

	def __init__(self, panoMeta, panorama):
		self.panoMeta = panoMeta
		self.panorama = panorama
		self.DecodeDepthMap(panoMeta['rawDepth'])

	def DecodeDepthMap(self, raw):
		raw = zlib.decompress(base64.urlsafe_b64decode(raw + self.MakePadding(raw)))

		pos = 0		

		(headerSize, numPlanes, panoWidth, panoHeight, planeIndicesOffset) = struct.unpack('<BHHHB', raw[0:8])
		self.DepthHeader = {'numPlanes' : numPlanes, 'panoWidth' : panoWidth, 'panoHeight' : panoHeight}
		
		if headerSize != 8 or planeIndicesOffset != 8:
			print ("Invalid depthmap data")
			return
		pos += headerSize

		self.DepthMapIndices = [x for x in raw[planeIndicesOffset:planeIndicesOffset + (panoWidth * panoHeight)]]
		pos += len(self.DepthMapIndices)

		self.DepthMapPlanes = []
		for i in range(0, numPlanes):
			(nx, ny, nz, d) = struct.unpack('<ffff', raw[pos:pos+16])

			self.DepthMapPlanes.append({ 'd': d, 'nx': nx, 'ny': ny, 'nz': nz }) # nx/ny/nz = unit normal, d = distance from origin
			pos += 16

	def MakePadding(self, s):
		return (4 - (len(s) % 4)) * '='
	def showDepth(self):
		depthMap = -self.depthMap * 255 / 50		
		depthMap[np.nonzero(np.isnan(depthMap))] = 255
		depthMap[np.nonzero(depthMap > 255)] = 255
		cv2.imshow('image', depthMap.astype(np.uint8))
		cv2.waitKey(0)
	def showIndex(self):
		height = self.DepthHeader['panoHeight']
		width = self.DepthHeader['panoWidth']
		indices= np.array((self.DepthMapIndices))
		indexMap = np.zeros((height * width, 3), dtype = np.uint8)
		colorMap = np.random.random_integers(0,255,(self.DepthHeader['numPlanes'], 3))
		for i in range(0,self.DepthHeader['numPlanes']):
			indexMap = np.zeros((height * width, 3), dtype = np.uint8)
			indexMap[np.nonzero(indices == i), :] = colorMap[i,:]
			cv2.imwrite('index'+ str(i) +'.png',indexMap.reshape((height, width, 3)))

		#indexMap *=  255 / 50	
		#indexMap[np.nonzero(indexMap > 255)] = 255
		cv2.imshow('image', indexMap.reshape((height, width, 3)).astype(np.uint8))
		cv2.waitKey(0)
	def CreatePtCloud(self, v):
		if self.DepthHeader['panoHeight'] != 256 or self.DepthHeader['panoWidth'] != 512:
			print (self.panoMeta['panoId'], 'DepthMap unsual')
			return
		height = self.DepthHeader['panoHeight']
		width = self.DepthHeader['panoWidth']
		depthMap = np.zeros([height, width], dtype = np.float32)
		panorama = cv2.resize(self.panorama, (width, height), cv2.INTER_LINEAR )
		data = np.zeros((height, width), dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])
		for h in range(0, height):
			for w in range(0, width):
				planeIndex = self.DepthMapIndices[h*width +w]
				plane = self.DepthMapPlanes[planeIndex]
				depth = plane['d'] / (v[h,w,:].dot(np.array([plane['nx'], plane['ny'], plane['nz']])))
				depthMap[h][w] = depth
				data[h][w]['a_position'] = depth * v[h,w,:]
				#data[h][w]['a_color'] = np.array(panorama[h,w]) / 255
		self.depthMap = depthMap		
		data['a_color'] = np.array(panorama) / 255
		self.data_ptCLoud = data	
	def CreatePtCloud2(self, v):
		height = self.DepthHeader['panoHeight']
		width = self.DepthHeader['panoWidth']
		depthMap = np.zeros((height*width), dtype = np.float32)
		planeIndices = np.array(self.DepthMapIndices)
		depthMap[np.nonzero(planeIndices == 0)] = np.nan
		v = v.reshape((height*width, 3))
		for i in range (1, self.DepthHeader['numPlanes']):
			plane = self.DepthMapPlanes[i]
			p_depth = np.ones((height*width)) * plane['d']
			depth = p_depth / v.dot(np.array((plane['nx'], plane['ny'], plane['nz'])))
			#depth = depth.reshape((height, width))
			depthMap[np.nonzero(planeIndices == i)] = depth[np.nonzero(planeIndices == i)]
			#depthMap = depth
		self.depthMap = depthMap.reshape((height, width))
		panorama = cv2.resize(self.panorama, (width, height), cv2.INTER_LINEAR )
		data = np.zeros((height, width), dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])
		xyz = ((np.transpose(v) * np.matlib.repmat(depthMap, 3,1))) 
		data['a_position'] = np.transpose(xyz).reshape((height, width, 3))
		data['a_color'] = np.array(panorama) / 255
		self.data_ptCLoud = data		
def CreateSphericalRay(height, width):
    	
    h = np.arange((height))
    theta = (height - h - 0.5) / height * np.pi
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    w = np.arange((width))
    phi = (width - w - 0.5) / width * 2 * np.pi + np.pi/2
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    v = np.zeros((height, width, 3))
	## interesting
    v[:,:,0] = sin_theta.reshape((height,1)) * cos_phi
    v[:,:,1] = sin_theta.reshape((height,1)) * sin_phi
    v[:,:,2] = cos_theta.reshape((height,1)) * np.ones(width)

    return v