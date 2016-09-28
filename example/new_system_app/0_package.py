#!/usr/bin/python3
# ==============================================================
# Pack the matrix process into google parse
# ==============================================================
import numpy as np
import json
import sys
import triangle

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting

# Create dashCamFileProcess and load 50 top Dashcam
needMatchInfo3d = False
dashCamFileProcess = file_process.DashCamFileProcessor()
gpyWindow = glumpy_setting.GpyWindow()
"""
For Visual
"""
sleIndex = 0
for fileIndex in range(sleIndex,sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)
    #fileID += '_info3d'

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

    """
    For info_3d
    """
    '''
    for img, gps in info_3d.items():
        for gps, indices in info_3d[img].items():
            print(gps, sv3DRegion.fileMeta['info2ID'][gps], sv3DRegion.fileMeta['id2GPS'][sv3DRegion.fileMeta['info2ID'][gps]])
            sv3D = sv3DRegion.sv3D_Dict[sv3DRegion.fileMeta['info2ID'][gps]]
            programSV3D = glumpy_setting.ProgramSV3D(data=sv3D.ptCLoudData, name=str(index), point_size=1)

            index += 1
    '''

programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data, name=None, point_size=1, anchor=anchor_matrix_whole)
gpyWindow.add_program(programSV3DRegion)

programAxis = glumpy_setting.ProgramAxis(line_length=5)
gpyWindow.add_program(programAxis)

gpyWindow.run()

