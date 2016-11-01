#!/usr/bin/python3
# ==============================================================
# Step.5
# Output the ply file of SV3D constructed from trajectory
# ==============================================================
import numpy as np
from plyfile import PlyData, PlyElement
import json
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting
import dashcam_parse


sleIndex = 3
createSFM = False
createTrajectory = False
createSV = True
needAlign = True
needMatchInfo3d = True
needVisual = False
needOutput = True
needGround = False
mapType = '_trajectory'

# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()
"""
Process the select file
"""
for fileIndex in range(47, 49+1):
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
    # anchor is sorted and pick the first one
    # load the anchor that had been stored
    anchor = dashCamFileProcess.get_trajectory_anchor(fileID)
    fileID += mapType
    sv3DRegion = google_parse.StreetView3DRegion(fileID)
    sv3DRegion.init_region(anchor=anchor)
    anchor_matrix_whole = sv3DRegion.anchorMatrix

    if createSV and not sv3DRegion.QQ:
        index = 0
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
            #if index > 10:
            #    break

        programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data, name='ProgramSV3DRegion',
                                                             point_size=1, anchor_matrix=anchor_matrix_whole)
        programSV3DRegionGnd = glumpy_setting.ProgramSV3DRegion(data=dataGnd, name='ProgramSV3DRegion',
                                                             point_size=1, anchor_matrix=anchor_matrix_whole)
        if needMatchInfo3d:
            programSV3DRegion.apply_anchor_flip()
            programSV3DRegionGnd.apply_anchor_flip()

        """
        ALL PLY EXPORT IN HERE
        """
        if createSV and needOutput:
            data = programSV3DRegion.data
            data['a_color'] *= 255
            data['a_color'].astype(int)

            dataPLY = np.zeros(len(data), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                                                 ('blue', 'u1')])
            dataPLY['x'] = data['a_position'][:, 0]
            dataPLY['y'] = data['a_position'][:, 1]
            dataPLY['z'] = data['a_position'][:, 2]

            dataPLY['red'] = data['a_color'][:, 0]
            dataPLY['green'] = data['a_color'][:, 1]
            dataPLY['blue'] = data['a_color'][:, 2]

            el = PlyElement.describe(dataPLY, 'vertex')

            # PlyData([el], text=True).write('137_4_ascii.ply')
            PlyData([el]).write('/home/andy/src/ply/' + fileID + '_non_ground_binary.ply')

            if needGround:
                dataGnd = programSV3DRegion.data
                dataGnd['a_color'] *= 255
                dataGnd['a_color'].astype(int)

                dataPLY = np.zeros(len(dataGnd), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
                dataPLY['x'] = dataGnd['a_position'][:, 0]
                dataPLY['y'] = dataGnd['a_position'][:, 1]
                dataPLY['z'] = dataGnd['a_position'][:, 2]

                dataPLY['red'] = dataGnd['a_color'][:, 0]
                dataPLY['green'] = dataGnd['a_color'][:, 1]
                dataPLY['blue'] = dataGnd['a_color'][:, 2]

                el = PlyElement.describe(dataPLY, 'vertex')

                # PlyData([el], text=True).write('137_4_ascii.ply')
                PlyData([el]).write(fileID + '_ground_binary.ply')


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
