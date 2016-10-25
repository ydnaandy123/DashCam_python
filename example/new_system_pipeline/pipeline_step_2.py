#!/usr/bin/python3
# ==============================================================
# Step.2
# Align the SFM with Google
# ==============================================================
import numpy as np
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting
import dashcam_parse


sleIndex = 5
createSFM = True
createSV = True
needMatchInfo3d = True
needVisual = True
mapType = '_info3d'
# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()
# Manual anchor, but I think this is so wrong.
#anchor = {'panoId': 'OAvT8QfoqjB1F6wVX747rw', 'Lat': 25.061674, 'Lon': 121.652461}
"""
Process the select file
"""
for fileIndex in range(sleIndex,sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)
    """
    Create the sfm point cloud
    """
    if createSFM:
        sfm3DRegion = dashcam_parse.SFM3DRegion(fileID)


    """
    Create the global metric point cloud,
    then set the region anchor
    """
    if createSV:
        fileID += mapType
        sv3DRegion = google_parse.StreetView3DRegion(fileID)
        sv3DRegion.init_region(anchor=None)
        anchor_matrix_whole = sv3DRegion.anchorMatrix

        index = 0
        for sv3D_id, sv3D in sorted(sv3DRegion.sv3D_Dict.items()):
            sv3D.apply_global_adjustment()
            sv3D.apply_local_adjustment()

            if index == 0:
                data = sv3D.ptCLoudData
            else:
                data = np.concatenate((data, sv3D.ptCLoudData), axis=0)

            index += 1
            #if index > 10:
            #    break


"""
For Visualize
"""
if needVisual:
    gpyWindow = glumpy_setting.GpyWindow()

    if createSFM:
        programSV3DRegion = glumpy_setting.ProgramSFM3DRegion(data=sfm3DRegion.ptcloudData, name='ProgramSFM3DRegion', point_size=1)
        gpyWindow.add_program(programSV3DRegion)

    if createSV:
        programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data, name=None, point_size=1, anchor_matrix=anchor_matrix_whole)
        if needMatchInfo3d:
            programSV3DRegion.apply_anchor()
        gpyWindow.add_program(programSV3DRegion)

    programAxis = glumpy_setting.ProgramAxis(line_length=5)
    gpyWindow.add_program(programAxis)

    gpyWindow.run()

