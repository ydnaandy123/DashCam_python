#!/usr/bin/python3
# ==============================================================
# get pano and depth from gps
# ==============================================================
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import google_store
import google_parse


# Create PanoFetcher
zoom, radius = 2, 10
fileID = 'NTHU'
"""
# 1. use BFS
# parameter: fileId, gps, queryNum
"""
if False:
    panoFetcher = google_store.PanoFetcher(zoom, radius)
    panoFetcher.bfs_aug(fileID, (24.7963071, 120.992373), 5)

"""
# 2. parse raw depth
"""
if True:
    sv3DRegion = google_parse.StreetView3DRegion(fileID)
    sv3DRegion.init_region(anchor=None)
    sv3DRegion.create_topoloy()
    sv3DRegion.create_region()
    for key in sv3DRegion.sv3D_Dict:
        sv = sv3DRegion.sv3D_Dict[key]
        #sv.visualize()
        break

"""
" 3. openGl visual
"""
import numpy as np
import glumpy_setting

needMatchInfo3d = True
needVisual = True
addPlane = False
needGround =False

if True:
    #pano_length = len(sv3DRegion.panoramaList)
    pano_length = len(sv3DRegion.sv3D_Dict)
    anchor_inv = np.linalg.inv(sv3DRegion.anchorMatrix)
    zero_vec, pano_ori_set, dis_len_set = [0, 0, 0, 1], np.zeros((pano_length, 3)), np.zeros(pano_length)
    # Initialize the pano according to location(lat, lon)
    for i in range(0, pano_length):
        sv3D = sv3DRegion.sv3D_Time[i]
        sv3D.apply_global_adjustment()  # Absolute position on earth (lat, lon, yaw)
        sv3D.apply_local_adjustment()  # Relative position according to anchor (anchor's lat,lon)
        if needMatchInfo3d:
            # This rotate the SV3D for matching x-y plane
            # Actually we build the point cloud on x-y plane
            # So we just multiply the inverse matrix of anchor
            sv3D.apply_anchor_adjustment(anchor_matrix=sv3DRegion.anchorMatrix)
        # Record all pano location(relative)
        # And find the nearest panorama
    isFirst = True
    for key in sv3DRegion.sv3D_Dict:
        sv3D = sv3DRegion.sv3D_Dict[key]
        if addPlane:
            sv3D.auto_plane()
        if isFirst:
            data = sv3D.ptCLoudData[np.nonzero(sv3D.non_con)]
            dataGnd = sv3D.ptCLoudData[np.nonzero(sv3D.gnd_con)]
            isFirst = False
        else:
            data = np.concatenate((data, sv3D.ptCLoudData[np.nonzero(sv3D.non_con)]), axis=0)
            dataGnd = np.concatenate((dataGnd, sv3D.ptCLoudData[np.nonzero(sv3D.gnd_con)]), axis=0)



    if needVisual:
        programSV3DRegion = glumpy_setting.ProgramSV3DRegion(
            data=data, name='ProgramSV3DRegion', point_size=1,
            anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d)
        programSV3DRegionGnd = glumpy_setting.ProgramSV3DRegion(
            data=dataGnd, name='ProgramSV3DRegionGnd', point_size=1,
            anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d)
        programSV3DTopology = glumpy_setting.ProgramSV3DTopology(
            data=sv3DRegion.topologyData, name='programSV3DTopology',
            anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d)

    """
    For Visualize
    """
    if needVisual:
        gpyWindow = glumpy_setting.GpyWindow()
        if addPlane:
            for j in range(0, pano_length):
                sv3D = sv3DRegion.sv3D_Time[j]
                for i in range(0, len(sv3D.all_plane)):
                    programGround = glumpy_setting.ProgramPlane(data=sv3D.all_plane[i]['data'], name='test',
                                                                face=sv3D.all_plane[i]['tri'])
                    gpyWindow.add_program(programGround)
        else:
            gpyWindow.add_program(programSV3DRegion)
            gpyWindow.add_program(programSV3DTopology)
            if needGround:
                gpyWindow.add_program(programSV3DRegionGnd)

        programAxis = glumpy_setting.ProgramAxis(line_length=5)
        gpyWindow.add_program(programAxis)

        gpyWindow.run()
