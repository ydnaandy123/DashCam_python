#!/usr/bin/python3
# ==============================================================
# Pack the matrix process into google parse
# ==============================================================
import numpy as np
import sys
import triangle
from scipy import spatial
from plyfile import PlyData, PlyElement

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting

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
    fileID += '_trajectory'

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
        #if index > 0:
        #    break
        #break


gpyWindow = glumpy_setting.GpyWindow()

programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data, name=None, point_size=1, anchor_matrix=anchor_matrix_whole)
programSV3DRegion.apply_anchor()
#gpyWindow.add_program(programSV3DRegion)

data = programSV3DRegion.data
"""
KD_tree
"""
'''
points = data['a_position'][::1000]
tree = spatial.KDTree(points)
test = 0
for pt in data['a_position']:
    print((tree.query_ball_point(x=pt, r=0.5)))
    test += 1
    if test > 50:
        break
'''
"""
Triangle
"""
tri = np.array(triangle.delaunay(data['a_position'][:, 0:2]), dtype=np.uint32)
#data['a_position'][:, 2] = 0
programGround = glumpy_setting.ProgramPlane(data=data, name=str(index), face=tri)
gpyWindow.add_program(programGround)
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
xyzzz['z'] = data['a_position'][:, 2]
xyzzz['red'] = data['a_color'][:, 0]
xyzzz['green'] = data['a_color'][:, 1]
xyzzz['blue'] = data['a_color'][:, 2]
el = PlyElement.describe(xyzzz, 'vertex')

face = np.zeros(len(tri), dtype=[('vertex_indices', 'i4', 3)])
face['vertex_indices'] = tri
tri = PlyElement.describe(face, 'face')

PlyData([el, tri], text=True).write('some_ascii.ply')
#PlyData([el]).write('over_simple_binary.ply')
'''

"""
RUN
"""
#programAxis = glumpy_setting.ProgramAxis(line_length=5)
#gpyWindow.add_program(programAxis)

gpyWindow.run()

