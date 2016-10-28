#!/usr/bin/python2
# ==============================================================
# Step.4
# Fetch the Google data
# According to trajectory!!
# ==============================================================
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_store

sleIndex = 1

# Create PanoFetcher
zoom, radius = 1, 10
panoFetcher = google_store.PanoFetcher(zoom, radius)

# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()

# Select one of the fileName among the 50 selected files
if __name__ == '__main__':
    for i in range(sleIndex, sleIndex+1):

        index = i
        fileID = str(dashCamFileProcess.list50[index][1])
        print(fileID, index)

        """
        # 1. use info_3d pathPoint
        """
        #pathPoint_set_info3d = dashCamFileProcess.get_path_info3d(file_id=fileID)
        #print(pathPoint_set_info3d)
        #panoFetcher.info_3d(fileID, pathPoint_set_info3d)

        """
        # 2. use BFS
        # Here use the first point in info_3d
        """
        #lat, lon = None, None
        #for pathPoint in pathPoint_set_info3d:
        #    print(pathPoint)
        #    [lat, lon] = pathPoint.split(',')
        #    break
        panoFetcher.bfs_aug('testttt', (24.797163,120.9965176), 10)

        """
        3. use info_3d plus BFS
        """
        # well... it's too big to handle now..
        # too bad :'(

        """
        4. use trajectory
        """
        #path_trajectory = dashCamFileProcess.get_path_trajectory(file_id=fileID)
        #panoFetcher.trajectory(fileID, path_trajectory)

