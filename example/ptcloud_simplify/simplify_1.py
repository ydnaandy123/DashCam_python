#!/usr/bin/python3
# ==============================================================
# Now let's add the ProjectionPanoYawDeg(same mean with heading I guess) back
# heading (default 0) defines the rotation angle around the camera locus in degrees relative from true north.
# Headings are measured clockwise (90 degrees is true east).
# ==============================================================
import numpy as np
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting


sleIndex = 3
createSV = True
needAlign = True
needMatchInfo3d = True
needVisual = True
needGround = False
mapType = '_trajectory'

# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()
"""
Process the select file
"""
for fileIndex in range(sleIndex, sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)

    """
    Create the global metric point cloud,
    then set the region anchor
    """
    # anchor is sorted and pick the first one
    # now load the anchor that had been stored
    anchor = dashCamFileProcess.get_trajectory_anchor(fileID)
    fileID += mapType
    sv3DRegion = google_parse.StreetView3DRegion(fileID)
    sv3DRegion.init_region(anchor=anchor)
    # why QQ?
    # because the map didn't query the anchor file
    if sv3DRegion.QQ:
        continue

    if createSV:
        index = 0
        sv3DRegion.create_topoloy()
        #sv3DRegion.create_single()
        sv3DRegion.create_region()
        for sv3D_id, sv3D in sorted(sv3DRegion.sv3D_Dict.items()):
            sv3D.apply_global_adjustment()  # Absolute position on earth
            sv3D.apply_local_adjustment()  # Relative position according to anchor

            if index == 0:
                 data = sv3D.ptCLoudData
                 dataGnd = sv3D.ptCLoudDataGnd
            else:
                 data = np.concatenate((data, sv3D.ptCLoudData), axis=0)
                 dataGnd = np.concatenate((data, sv3D.ptCLoudDataGnd), axis=0)

            index += 1

        programSV3DRegion = glumpy_setting.ProgramSV3DRegion(
            data=data, name='ProgramSV3DRegion',
            anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw)
        programSV3DRegionGnd = glumpy_setting.ProgramSV3DRegion(
            data=dataGnd, name='ProgramSV3DRegion',
            anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw)
        programSV3DTopology = glumpy_setting.ProgramSV3DTopology(
            data=sv3DRegion.topologyData, name='ProgramSV3DRegion',
            anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw)
        if needMatchInfo3d:
            programSV3DRegion.apply_anchor_flip()
            programSV3DRegionGnd.apply_anchor_flip()
            programSV3DTopology.apply_anchor_flip()


"""
For Visualize
"""
if needVisual:
    gpyWindow = glumpy_setting.GpyWindow()

    if createSV:
        gpyWindow.add_program(programSV3DRegion)
        gpyWindow.add_program(programSV3DTopology)
        if needGround:
            gpyWindow.add_program(programSV3DRegionGnd)

    programAxis = glumpy_setting.ProgramAxis(line_length=5)
    gpyWindow.add_program(programAxis)

    gpyWindow.run()
