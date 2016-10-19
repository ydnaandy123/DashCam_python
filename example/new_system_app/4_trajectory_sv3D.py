#!/usr/bin/python3
# ==============================================================
# Pack the matrix process into google parse
# ==============================================================
import numpy as np
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting

# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()
"""
For Visual
"""
sleIndex = 3
for fileIndex in range(sleIndex,sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)
    fileID += '_trajectory'

    """
    Create the global metric point cloud,
    then set the region anchor
    """
    sv3DRegion = google_parse.StreetView3DRegion(fileID)
    sv3DRegion.init_region(anchor=None)

    anchor_matrix_whole = np.eye(4, dtype=np.float32)

    index = 0
    for sv3D_id, sv3D in sv3DRegion.sv3D_Dict.items():
        sv3D.apply_global_adjustment()
        sv3D.apply_local_adjustment()

        if index == 0:
            anchor_matrix_whole = np.dot(sv3D.matrix_local, sv3D.matrix_global)
            data = sv3D.ptCLoudData
        else:
            data = np.concatenate((data, sv3D.ptCLoudData), axis=0)

        index += 1
        if index > 5:
            break
        #break


gpyWindow = glumpy_setting.GpyWindow()

programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data, name=None, point_size=1, anchor_matrix=anchor_matrix_whole)
gpyWindow.add_program(programSV3DRegion)

programAxis = glumpy_setting.ProgramAxis(line_length=5)
gpyWindow.add_program(programAxis)

gpyWindow.run()

