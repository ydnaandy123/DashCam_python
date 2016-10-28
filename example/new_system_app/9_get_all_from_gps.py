#!/usr/bin/python2
# ==============================================================
# get pano and depth from gps
# ==============================================================
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
#import google_store
import google_parse


# Create PanoFetcher
zoom, radius = 1, 10
fileID = 'testttt'
"""
# 1. use BFS
# parameter: fileId, gps, queryNum
"""
#panoFetcher = google_store.PanoFetcher(zoom, radius)
#panoFetcher.bfs_aug(fileID, (24.797163,120.9965176), 1)

"""
# 2. parse raw depth
"""
sv3DRegion = google_parse.StreetView3DRegion(fileID)
sv3DRegion.init_region(anchor=None)
