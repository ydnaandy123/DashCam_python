#!/usr/bin/python3
# ==============================================================
# Pack the matrix process into google parse
# ==============================================================
import numpy as np
import triangle
from glumpy import glm
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting
import base_process

# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()
# Manual anchor, but I think this is so wrong.
#anchor = {'panoId': 'uSjqj9Lt256V8I7RckMykA', 'Lat': 25.068939, 'Lon': 121.479781}
anchor = {'panoId': 'JfAAg1RD0myOqNIU0utdNA', 'Lat': 22.622543, 'Lon': 120.285735}

"""
For Visual
"""
sleIndex = 6
for fileIndex in range(sleIndex,sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)
    fileID += '_info3d'

    """
    Create the global metric point cloud,
    then set the region anchor
    """
    sv3DRegion = google_parse.StreetView3DRegion(fileID)
    sv3DRegion.init_region(anchor=None)

    anchor_matrix_whole = sv3DRegion.anchorMatrix

    index = 0
    for sv3D_id, sv3D in sorted(sv3DRegion.sv3D_Dict.items()):

        ### WHY???
        sv3D.create_ptcloud_ground_grid()

        sv3D.apply_global_adjustment()
        sv3D.apply_local_adjustment()

        if index == 0:
            data = sv3D.ptCLoudData
            data_gnd = sv3D.ptCLoudDataGnd
            data_gnd_grid = sv3D.ptCLoudDataGndGrid
        else:
            data = np.concatenate((data, sv3D.ptCLoudData), axis=0)
            data_gnd = np.concatenate((data_gnd, sv3D.ptCLoudDataGnd), axis=0)
            data_gnd_grid = np.concatenate((data_gnd_grid, sv3D.ptCLoudDataGndGrid), axis=0)

        index += 1
        if index > 0:
            break
        #break


gpyWindow = glumpy_setting.GpyWindow()

#programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data, name=None, point_size=1, anchor_matrix=anchor_matrix_whole)
#programSV3DRegion.apply_anchor()
#gpyWindow.add_program(programSV3DRegion)

programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data_gnd, name=None, point_size=1, anchor_matrix=anchor_matrix_whole)
programSV3DRegion.apply_anchor()
gpyWindow.add_program(programSV3DRegion)

#programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data_gnd_grid, name=None, point_size=1, anchor_matrix=anchor_matrix_whole, alpha=0)
#programSV3DRegion.apply_anchor()
#gpyWindow.add_program(programSV3DRegion)

"""
Triangle
"""
tri = np.array(triangle.delaunay(data_gnd['a_position'][:, 0:2]), dtype=np.uint32)
#data_gnd_grid['a_position'] = base_process.sv3d_apply_m4(data=data_gnd_grid['a_position'], m4=np.linalg.inv(anchor_matrix_whole))
#data_gnd['a_position'][:, 2] = 0
programGround = glumpy_setting.ProgramPlane(data=data_gnd, name=str(index), face=tri)
gpyWindow.add_program(programGround)

#programAxis = glumpy_setting.ProgramAxis(line_length=5)
#gpyWindow.add_program(programAxis)

gpyWindow.run()

