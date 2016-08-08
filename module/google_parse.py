#!/usr/bin/python
import numpy as np
import cv2
import zlib
import base64
import struct
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
				if np.isnan(depth):
					depthMap[h][w] = np.nan
				else:
					depthMap[h][w] = -depth / 50 * 255
				data[h][w]['a_position'] = depth * v[h,w,:]
				#data[h][w]['a_color'] = np.array(panorama[h,w]) / 255
			
		depthMap[depthMap == np.nan] = 255
		cv2.imshow('image', depthMap.astype(np.uint8))
		cv2.waitKey(0)
		data['a_color'] = np.array(panorama) / 255
		self.data_ptCLoud = data	

def CreateSphericalRay(height, width):
    	
    h = np.arange(height)
    theta = (height - h - 0.5) / height * np.pi
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    w = np.arange(width)
    phi = (width - w - 0.5) / width * 2 * np.pi + np.pi/2
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    v = np.zeros((height, width, 3))
    v[:,:,0] = sin_theta.reshape(height,1) * cos_phi
    v[:,:,1] = sin_theta.reshape(height,1) * sin_phi
    v[:,:,2] = cos_theta.reshape(height,1) * np.ones(width)

    return v