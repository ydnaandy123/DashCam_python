#!/usr/bin/python2
import sys
sys.path.append('module')	# use the module under 'module'
import streetview_my
import numpy as np 
import base_process

lat, lon = 25.0446577, 121.5384595
print base_process.GPS2GL_greatCircle(lat, lon)
print base_process.GL2GPS_greatCircle(base_process.GPS2GL_greatCircle(lat, lon))

[X, Y, Z] = base_process.geo2ECEF(lat = lat, lon = lon)
print (Y, Z, X) 
print base_process.ECEF2geo(X, Y, Z)
#google_store.BFS('Civic_Boulevard', (25.0446577,121.5384595), 50)
