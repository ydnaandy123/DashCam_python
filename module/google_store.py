#!/usr/bin/python2
import numpy as np
import cv2
import urllib
import urllib2
import libxml2
import json
import os

# pathPoint in different form
class PanoFetcher:
	def __init__(self, zoom = 1, radius = 30):
		self.zoom = zoom		# Get what pano
		self.radius = radius	# Panorama's' size	

	def fileDirCheck(self, fileID):
		if not os.path.exists('src/panometa/' + fileID):
			os.makedirs('src/panometa/' + fileID)       
		if not os.path.exists('src/panorama/' + fileID):					
			os.makedirs('src/panorama/' + fileID)      	

	def BFS(self, fileID, startGPS, maxPano=100):
		self.fileDirCheck(fileID)
		panoSet = set()
		panoList = []
		panoDict = {}
		cur = 0
		# Change the GPS to panoID
		# and put fisrt panoID into the list
		(lat, lon) = startGPS
		pano = getPanoramaMetadata(lat=lat, lon=lon, radius=self.radius)
		panoSet.add(pano.PanoId)
		panoList.append(pano.PanoId)	
		# Until maximum
		for cur in range(0, maxPano):
    		# Get the pano accroding to the list 
			pano = getPanoramaMetadata(panoid = panoList[cur], radius=self.radius)
			panoDict[pano.PanoId] = [pano.Lat, pano.Lon]	
			img = getPanorama(pano.PanoId,  zoom = self.zoom)
			try:
				pano_basic = {'panoId':pano.PanoId, 'Lat':pano.Lat, 
							'Lon':pano.Lon, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg, 
							'AnnotationLinks':pano.AnnotationLinks, 'rawDepth':pano.rawDepth, 
							'Text':pano.Text}
				# Add new founded panos into the list
				for link in pano.AnnotationLinks:	
					if (link['PanoId'] not in panoSet):				
						panoId = link['PanoId']
						panoList.append(panoId)
						panoSet.add(panoId)						
			except:
				print pano.PanoId + ' is file corrupt.'
				pano_basic = {}
			with open('src/panometa/' + fileID + '/' + pano.PanoId + '.json', 'w') as outfile:
				json.dump(pano_basic, outfile)
				outfile.close()						
			cv2.imwrite('src/panorama/' + fileID + '/' + pano.PanoId + '.jpg', img)
			print cur, pano.Lat, pano.Lon, pano.PanoId
		with open('src/panometa/' + fileID + '/fileMeta.json', 'w') as outfile:
			fileMeta = {'panoList':panoList, 'cur':cur, 'id2GPS':panoDict}
			print "id2GPS's length: %d" % len (panoDict)
			print "panoList's length: %d" % len(panoList)
			json.dump(fileMeta, outfile)
			outfile.close()								
		return				
	def BFS_aug(self, fileID, startGPS=None, maxPano=100):		
		fname = 'src/panometa/' + fileID + '/fileMeta.json'
		if os.path.isfile(fname) :
			print 'Augment the existing region"' + fileID + '"(accroding to the fileMeta):'
			with open(fname) as data_file:    
				fileMeta = json.load(data_file)
				panoList = fileMeta['panoList']
				cur = fileMeta['cur']
				panoDict = fileMeta['id2GPS']
				print panoList, cur
				data_file.close() 

			panoSet = set(panoList)
			cur = cur + 1
			# Until maximum
			for cur in range(cur + 0, cur + maxPano):
				# Get the pano accroding to the list 
				pano = getPanoramaMetadata(panoid = panoList[cur], radius=self.radius)
				panoDict[pano.PanoId] = [pano.Lat, pano.Lon]	
				img = getPanorama(pano.PanoId,  zoom = self.zoom)
				try:
					pano_basic = {'panoId':pano.PanoId, 'Lat':pano.Lat, 
								'Lon':pano.Lon, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg, 
								'AnnotationLinks':pano.AnnotationLinks, 'rawDepth':pano.rawDepth, 
								'Text':pano.Text}
					# Add new founded panos into the list
					for link in pano.AnnotationLinks:	
						if (link['PanoId'] not in panoSet):				
							panoId = link['PanoId']
							panoList.append(panoId)
							panoSet.add(panoId)						
				except:
					print pano.PanoId + ' is file corrupt.'
					pano_basic = {}
				with open('src/panometa/' + fileID + '/' + pano.PanoId + '.json', 'w') as outfile:
					json.dump(pano_basic, outfile)
					outfile.close()			
				cv2.imwrite('src/panorama/' + fileID + '/' + pano.PanoId + '.jpg', img)
				print cur, pano.Lat, pano.Lon, pano.PanoId	
			with open('src/panometa/' + fileID + '/fileMeta.json', 'w') as outfile:
				fileMeta = {'panoList':panoList, 'cur':cur, 'id2GPS':panoDict}
				print "id2GPS's length: %d" % len (panoDict)
				print "panoList's length: %d" % len(panoList)
				json.dump(fileMeta, outfile)
				outfile.close()								
			return
		else:
			print 'Create the region"' + fileID + '" first time.(Or lack of fileMeta):'
			self.BFS(fileID, startGPS, maxPano)
			return		

	def info_3d(self, fileID, pathPoint_set_info3d):
		panoSet = set()
		fileID += '_info3d'
		self.fileDirCheck(fileID)	
		for pathPoint in pathPoint_set_info3d:
			[lat, lon] = pathPoint.split(',')
			pano = getPanoramaMetadata(lat=lat, lon=lon, radius=self.radius)
			print lat, lon, pano.PanoId
			if pano.PanoId not in panoSet:
				panoSet.add(pano.PanoId)
				img = getPanorama(pano.PanoId, zoom = self.zoom)
				try:
					pano_basic = {'panoId':pano.PanoId, 'Lat':pano.Lat, 
								'Lon':pano.Lon, 'ProjectionPanoYawDeg':pano.ProjectionPanoYawDeg, 
								'AnnotationLinks':pano.AnnotationLinks, 'rawDepth':pano.rawDepth, 
								'Text':pano.Text}
				except:
					print pano.PanoId + ' is file corrupt.'
					pano_basic = {}
				with open('src/panometa/' + fileID + '/' + pano.PanoId + '.json', 'w') as outfile:
					json.dump(pano_basic, outfile)
					outfile.close()
				cv2.imwrite('src/panorama/' + fileID + '/' + pano.PanoId + '.jpg', img)	

def getUrlContents(url):
	f = urllib2.urlopen(url)
	data = f.read()
	f.close()
	return data

# panoid is the value from panorama metadata
# OR: supply lat/lon/radius to find the nearest pano to lat/lon within radius
def getPanoramaMetadata(panoid = None, lat = None, lon = None, radius = 30):
	BaseUri = 'http://maps.google.com/cbk';
	url =  '%s?'
	url += 'output=xml'			# metadata output
	url += '&v=4'				# version
	url += '&dm=1'				# include depth map
	url += '&pm=1'				# include pano map
	if panoid == None:
		url += '&ll=%s,%s'		# lat/lon to search at
		url += '&radius=%s'		# search radius
		url = url % (BaseUri, lat, lon, radius)
	else:
		url += '&panoid=%s'		# panoid to retrieve
		url = url % (BaseUri, panoid)
	
	findpanoxml = getUrlContents(url)
	if not findpanoxml.find('data_properties'):
		return None
	return PanoramaMetadata(libxml2.parseDoc(findpanoxml))

class PanoramaMetadata:
    	
	def __init__(self, panodoc):
		try:
			self.PanoDoc = panodoc
			panoDocCtx = self.PanoDoc.xpathNewContext()

			self.PanoId = panoDocCtx.xpathEval("/panorama/data_properties/@pano_id")[0].content
			self.ImageWidth = panoDocCtx.xpathEval("/panorama/data_properties/@image_width")[0].content
			self.ImageHeight = panoDocCtx.xpathEval("/panorama/data_properties/@image_height")[0].content
			self.TileWidth = panoDocCtx.xpathEval("/panorama/data_properties/@tile_width")[0].content
			self.TileHeight = panoDocCtx.xpathEval("/panorama/data_properties/@tile_height")[0].content
			self.NumZoomLevels = panoDocCtx.xpathEval("/panorama/data_properties/@num_zoom_levels")[0].content
			self.Lat = panoDocCtx.xpathEval("/panorama/data_properties/@lat")[0].content
			self.Lon = panoDocCtx.xpathEval("/panorama/data_properties/@lng")[0].content
			self.OriginalLat = panoDocCtx.xpathEval("/panorama/data_properties/@original_lat")[0].content
			self.OriginalLon = panoDocCtx.xpathEval("/panorama/data_properties/@original_lng")[0].content
			#self.Copyright = panoDocCtx.xpathEval("/panorama/data_properties/copyright/text()")[0].content
			# some panorama hasn't the files follow
			# which will cause error
			try:
				self.Text = panoDocCtx.xpathEval("/panorama/data_properties/text/text()")[0].content
			except:
				self.Text = ''
			#self.Region = panoDocCtx.xpathEval("/panorama/data_properties/region/text()")[0].content
			#self.Country = panoDocCtx.xpathEval("/panorama/data_properties/country/text()")[0].content

			self.ProjectionType = panoDocCtx.xpathEval("/panorama/projection_properties/@projection_type")[0].content
			self.ProjectionPanoYawDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@pano_yaw_deg")[0].content
			self.ProjectionTiltYawDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@tilt_yaw_deg")[0].content
			self.ProjectionTiltPitchDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@tilt_pitch_deg")[0].content
			
			self.AnnotationLinks = []
			for cur in panoDocCtx.xpathEval("/panorama/annotation_properties/link"):			
				self.AnnotationLinks.append({ 'YawDeg': cur.xpathEval("@yaw_deg")[0].content,
							'PanoId': cur.xpathEval("@pano_id")[0].content,
							'RoadARGB': cur.xpathEval("@road_argb")[0].content
							# ,'Text': text
							# some panorama hasn't this file
							# which will cause error
							# text = cur.xpathEval("link_text/text()")[0].content
				})
			
			tmp = panoDocCtx.xpathEval("/panorama/model/depth_map/text()")
			if len(tmp) > 0:
				self.rawDepth = tmp[0].content
		except:
			pass


	def __str__(self):
		tmp = ''
		for x in inspect.getmembers(self):
			if x[0].startswith("__") or inspect.ismethod(x[1]):
				continue
			
			tmp += "%s: %s\n" % x
		return tmp
# h_base, w_base should be renamed...
def getPanorama(panoid, zoom):
	h = pow(2,zoom-1)
	w = pow(2, zoom)
	h_base = 416
	w_base = 512
	panorama = np.zeros((h*w_base, w*w_base, 3), dtype="uint8")
	for y in range(0,h):
		for x in range(0,w):
			img = getPanoramaTile(panoid, zoom, x, y)
			panorama[y*w_base:y*w_base+w_base, x*w_base:x*w_base+w_base, :] = img[0:w_base, 0:w_base, :]
	return panorama[0:h*h_base, 0:w*h_base, :]
	
    		
# panoid is the value from the panorama metadata
# zoom range is 0->NumZoomLevels inclusively
# x/y range is 0->?
def getPanoramaTile(panoid, zoom, x, y):
	BaseUri = 'http://maps.google.com/cbk';
	url =  '%s?'
	url += 'output=tile'			# tile output
	url += '&panoid=%s'			# panoid to retrieve
	url += '&zoom=%s'			# zoom level of tile
	url += '&x=%i'				# x position of tile
	url += '&y=%i'				# y position of tile
	url += '&fover=2'			# ???
	url += '&onerr=3'			# ???
	url += '&renderer=spherical'		# standard speherical projection
	url += '&v=4'				# version
	url = url % (BaseUri, panoid, zoom, x, y)
	return url_to_image(url)  
    
# Method #1: OpenCV, NumPy, and urllib
# FROM http://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR) 
	# return the image
	return image    	