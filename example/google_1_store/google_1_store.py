#!/usr/bin/python2
import sys
sys.path.append('module')	# use the module under 'module'
import file_process
import google_store

zoom, radius = 1, 300
pf = google_store.PanoFetcher(zoom, radius)
dcfp = file_process.DashCamFileProcessor()
dcfp.loadList50()
fileID = str(dcfp.list50[0][1])

#pathPoint_set_info3d = dcfp.getPath_info3d(fileID = fileID)
#print pathPoint_set_info3d
#pf.info_3d(fileID, pathPoint_set_info3d)


#pf.BFS('Civic_Boulevard', (25.0446577,121.5384595), 5000)
#pf.BFS('NTHU', (24.7947302,120.9910429), 20)
