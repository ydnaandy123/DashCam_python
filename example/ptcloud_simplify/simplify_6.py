#!/usr/bin/python3
# ==============================================================
# TODO: decimate
# TODO: plane register
# ==============================================================
import numpy as np
import triangle
import sys
import scipy.misc
import math

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting
import base_process


sleIndex = 0
createSV = True
needMatchInfo3d = True
needVisual = True
needGround = False
mapType = '_info3d' # [_info3d, _trajectory]

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
            sv3D.apply_global_adjustment()  # Absolute position on earth (lat, lon, yaw)
            sv3D.apply_local_adjustment()  # Relative position according to anchor (anchor's lat,lon)

            if index == 0:
                 data = sv3D.ptCLoudData
                 # TODO: pano
                 pano = sv3D.panorama
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
            # This rotate the SV3D for matching x-y plane
            # Actually we build the point cloud on x-y plane
            # So we just multiply the inverse matrix of anchor
            programSV3DRegion.apply_anchor_flip()
            programSV3DRegionGnd.apply_anchor_flip()
            programSV3DTopology.apply_anchor_flip()

"""
Manual boxing
"""
'''
xmin, xmax = -10, 10
ymin, ymax = -20, 20

nx, ny = (50, 210)
size = nx * ny

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
xv, yv = np.meshgrid(x, y)

test = np.dstack((xv, yv))
plane = np.zeros((ny, nx, 3), dtype=np.float32)
plane[:, :, 0:2] = test
plane[:, :, 2] = -2
plane = np.reshape(plane, (size, 3))

pano_copy = np.copy(pano)

color = np.zeros((size, 3), dtype=np.float32)
for y in range(0, ny):
    for x in range(0, nx):
        index = x + y*nx
        pos = plane[index] # (416, 832, 3)(0~255)
        lat, lon = base_process.gl_2_ecef_great_circle(pos)
        if pos[0] == 0 or pos[1] == 0:
            color[index] = [1, 0, 0]
        else:
            img_y = math.floor((-lat + 90) / 180 * 416)
            img_x = math.floor((lon + 180) / 360 * 832)
            color[index] = pano[img_y, img_x, :] / 255
            pano_copy[img_y-3:img_y+3, img_x-3:img_x+3, :] = [255, 0, 0]
            #print(pos, lat, lon, img_y, img_x)
            #scipy.misc.imshow(pano)


#scipy.misc.imshow(pano_copy)
data = np.zeros((nx*ny), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
data['a_position'] = plane
data['a_color'] = color

#I = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
tri = np.array(triangle.delaunay(data['a_position'][:, 0:2]), dtype=np.uint32)
'''


"""
For Visualize
"""
if needVisual:
    gpyWindow = glumpy_setting.GpyWindow()

    for i in range(0, len(sv3D.gnd_plane)):
        programGround = glumpy_setting.ProgramPlane(data=sv3D.gnd_plane[i]['data'], name='test',
                                                    face=sv3D.gnd_plane[i]['tri'])
        gpyWindow.add_program(programGround)

    if createSV:
        gpyWindow.add_program(programSV3DRegion)
        gpyWindow.add_program(programSV3DTopology)
        if needGround:
            gpyWindow.add_program(programSV3DRegionGnd)

    programAxis = glumpy_setting.ProgramAxis(line_length=5)
    gpyWindow.add_program(programAxis)

    gpyWindow.run()
