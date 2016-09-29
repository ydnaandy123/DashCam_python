#!/usr/bin/python3
# ==============================================================
# store pano base on trajectory
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

sleIndex = 3
for fileIndex in range(sleIndex,sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)

    pathPoint_set_trajectory = dashCamFileProcess.get_path_trajectory(file_id=fileID)
    print(pathPoint_set_trajectory)
    panoFetcher.trajectory(fileID, pathPoint_set_trajectory)

