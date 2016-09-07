#!/usr/bin/python2
import sys
sys.path.append('module')	# use the module under 'module'
import file_process
import google_store
import base_process

# Q1: google (lat, lon)
pano = google_store.getPanoramaMetadata(lat = 25.044670, lon = 121.538450)
print pano.Lat, pano.Lon
print pano.OriginalLat, pano.OriginalLon

# Q2: The World Geodetic System (WGS) is a standard 
#     or use in cartography, geodesy, and navigation including by GPS. 
#     It comprises a standard coordinate system for the Earth
lat, lon = 25.0446577, 121.5384595
# Original
print 'Original:'
print base_process.GPS2GL_greatCircle(lat, lon)
print base_process.GL2GPS_greatCircle(base_process.GPS2GL_greatCircle(lat, lon))
# ECEF + WGS84
print 'ECEF + WGS84:'
[X, Y, Z] = base_process.geo2ECEF(lat = lat, lon = lon)
print (Y, Z, X) 
print base_process.ECEF2geo(X, Y, Z)

# D1: BFS_aug
#in3d.png(multiple ways)
#google_1_store.py

#---python3---(local)

# D2: topology
#OSM
#google glumpy_4_common_setting

# D3: ptCloud construct time
#glumpy_1_drawPtCloud_streetview

# D4: kd-tree
#

# lack of demo...
#senmantic segment
#generate random viewPoint
#car on the street