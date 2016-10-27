#!/usr/bin/python3
# ==============================================================
# Step.4
# Output the trajectory in (lat, lon) form
# ==============================================================
import numpy as np
import json
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting
import dashcam_parse
import base_process


sleIndex = 21
createSFM = False
createTrajectory = False
createSV = True
needAlign = True
needMatchInfo3d = True
needVisual = True
mapType = '_info3d'
# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()
# anchor is sorted and pick the first one
"""
Process the select file
"""
for fileIndex in range(sleIndex,sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)
    """
    Create the sfm point cloud
    """
    sfm3DRegion = dashcam_parse.SFM3DRegion(fileID)
    if createSFM:
        programSFM3DRegion = glumpy_setting.ProgramSFM3DRegion(data=sfm3DRegion.ptcloudData, name='ProgramSFM3DRegion',
                                                               point_size=1, matrix=sfm3DRegion.matrix)
        if needAlign:
            programSFM3DRegion.align_flip()
    """
    Create the trajectory
    """
    if createTrajectory:
        programTrajectory = glumpy_setting.programTrajectory(data=sfm3DRegion.trajectoryData, name='programTrajectory',
                                                             point_size=6, matrix=sfm3DRegion.matrix)
        if needAlign:
            programTrajectory.align_flip()

    """
    Create the global metric point cloud,
    then set the region anchor
    """
    fileID += mapType
    sv3DRegion = google_parse.StreetView3DRegion(fileID)
    sv3DRegion.init_region(anchor=None)
    anchor_matrix_whole = sv3DRegion.anchorMatrix

    if createSV:
        index = 0
        for sv3D_id, sv3D in sorted(sv3DRegion.sv3D_Dict.items()):
            sv3D.apply_global_adjustment()  # Absolute position on earth
            sv3D.apply_local_adjustment()  # Relative position according to anchor

            if index == 0:
                data = sv3D.ptCLoudData
            else:
                data = np.concatenate((data, sv3D.ptCLoudData), axis=0)

            index += 1
            #if index > 10:
            #    break

        programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data, name='ProgramSV3DRegion',
                                                             point_size=1, anchor_matrix=anchor_matrix_whole)
        if needMatchInfo3d:
            programSV3DRegion.apply_anchor_flip()
"""
For Visualize
"""
if needVisual:
    gpyWindow = glumpy_setting.GpyWindow()

    if createSFM:
        gpyWindow.add_program(programSFM3DRegion)

    if createTrajectory:
        gpyWindow.add_program(programTrajectory)

    if createSV:
        gpyWindow.add_program(programSV3DRegion)

    programAxis = glumpy_setting.ProgramAxis(line_length=5)
    gpyWindow.add_program(programAxis)

    gpyWindow.run()

