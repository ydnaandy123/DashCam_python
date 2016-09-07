#!/usr/bin/python2
import sys
sys.path.append('module')	# use the module under 'module'
import file_process
import google_store

# Create PanoFetcher and dashCamFileProcess
zoom, radius = 1, 30
panofetcher = google_store.PanoFetcher(zoom, radius)
dashCamFileProcess = file_process.DashCamFileProcessor()

# Select one of the fileName
# among the 50 selected files
dashCamFileProcess.loadList50()
fileID = str(dashCamFileProcess.list50[10][1])

# 1.DashCam info_3d
# Get the (lat, lon) from info_3d
# Then fetch the corresponding panofile
#pathPoint_set_info3d = dashCamFileProcess.getPath_info3d(fileID = fileID)
#print pathPoint_set_info3d
#panofetcher.info_3d(fileID, pathPoint_set_info3d)


panofetcher.BFS_aug('test2', (24.7947302,120.9910429), 10)
#panofetcher.BFS('NTHU', (24.7947302,120.9910429), 20)
