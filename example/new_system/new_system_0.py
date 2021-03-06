#!/usr/bin/python2
# ==============================================================
# Showing how to require and store a 'large area' panaMeta from google
# accroding to the Dashcam 'info_3d'
# ==============================================================
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_store

# Create PanoFetcher
zoom, radius = 1, 30
panoFetcher = google_store.PanoFetcher(zoom, radius)

# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()

# Select one of the fileName among the 50 selected files
index = 8
fileID = str(dashCamFileProcess.list50[index][1])
print(fileID, index)


"""
# 1. use info_3d pathPoint
"""
pathPoint_set_info3d = dashCamFileProcess.get_path_info3d(file_id=fileID)
print(pathPoint_set_info3d)
panoFetcher.info_3d(fileID, pathPoint_set_info3d)

"""
# 2. use BFS
# Here use the first point in info_3d
"""
lat, lon = None, None
for pathPoint in pathPoint_set_info3d:
    print(pathPoint)
    [lat, lon] = pathPoint.split(',')
    break
panoFetcher.bfs_aug(fileID, (lat, lon), 5)

"""
3. use info_3d plus BFS
"""

