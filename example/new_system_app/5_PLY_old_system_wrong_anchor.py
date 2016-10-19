#!/usr/bin/python3
# ==============================================================
# Pack the matrix process into google parse
# ==============================================================
import numpy as np
import sys
import triangle
from plyfile import PlyData, PlyElement

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting

# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()
# Manual anchor, but I think this is so wrong.
anchor = {'panoId': 'JfAAg1RD0myOqNIU0utdNA', 'Lat': 22.622543, 'Lon': 120.285735}
#anchor = {'panoId': '_RAj8Tpy0wDG-5kGbhTwjA', 'Lat': 23.962967, 'Lon': 120.964846}
# 137 'JfAAg1RD0myOqNIU0utdNA', '22.622543', '120.285735'
# 731 '_RAj8Tpy0wDG-5kGbhTwjA', '23.962967', '120.964846'
"""
For Visual
"""
sleIndex = 0
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
        sv3D.apply_global_adjustment()
        sv3D.apply_local_adjustment()

        if index == 0:
            data = sv3D.ptCLoudData
        else:
            data = np.concatenate((data, sv3D.ptCLoudData), axis=0)

        index += 1
        #if index > 10:
        #    break
        #break


gpyWindow = glumpy_setting.GpyWindow()

programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data, name=None, point_size=1, anchor_matrix=anchor_matrix_whole)
programSV3DRegion.apply_anchor()

data = programSV3DRegion.data

"""
ALL PLY EXPORT IN HERE
"""
'''
data = programSV3DRegion.data
data['a_color'] *= 255
data['a_color'].astype(int)

xyzzz = np.zeros(len(data), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
xyzzz['x'] = data['a_position'][:, 0]
xyzzz['y'] = data['a_position'][:, 1]

data['a_position'][:, 2] = -2

xyzzz['z'] = data['a_position'][:, 2]
xyzzz['red'] = data['a_color'][:, 0]
xyzzz['green'] = data['a_color'][:, 1]
xyzzz['blue'] = data['a_color'][:, 2]
el = PlyElement.describe(xyzzz, 'vertex')

#PlyData([el], text=True).write('137_4_ascii.ply')
PlyData([el]).write('731_ground_binary.ply')
'''
data = programSV3DRegion.data
tri = np.array(triangle.delaunay(data['a_position'][:, 0:2]), dtype=np.uint32)
data['a_position'][:, 2] = -2
programGround = glumpy_setting.ProgramPlane(data=data, name=str(index), face=tri)
gpyWindow.add_program(programGround)

#gpyWindow.add_program(programSV3DRegion)

programAxis = glumpy_setting.ProgramAxis(line_length=5)
gpyWindow.add_program(programAxis)

gpyWindow.run()

