#!/usr/bin/python3
# ==============================================================
# Step.1
# Transform the point in SFM to Google (SV3D)
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
data_info, sleIndex = [], 1
for fileIndex in range(sleIndex,sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)

    with open('/home/andy/src/DashCam/json/newSystem_deep_match/' + fileID + '/info_3d.json') as data_file:
        info_3d = json.load(data_file)
        data_file.close()
    fileID += '_info3d'

    sv3DRegion = google_parse.StreetView3DRegion(fileID)

    index, ecef_offs, matrix_offs, pano_offs = 0, np.zeros(3), np.eye(4, dtype=np.float32), ''

    matrix_offs_whole = np.eye(4, dtype=np.float32)
    # ground
    for img, gps in info_3d.items():
        for gps, indices in info_3d[img].items():
            print(gps, sv3DRegion.fileMeta['info2ID'][gps], sv3DRegion.fileMeta['id2GPS'][sv3DRegion.fileMeta['info2ID'][gps]])
            sv3D = sv3DRegion.sv3D_Dict[sv3DRegion.fileMeta['info2ID'][gps]]
            programSV3D = glumpy_setting.ProgramSV3D(data=sv3D.ptCLoudData, name=str(index), point_size=1)
            groundPoint_cur = sv3D.ptCLoudData[(sv3D.ptCLoudData['a_position'][:, :, 2] > 2) * (sv3D.ptCLoudData['a_position'][:, :, 0] > 0)] #* (sv3D.ptCLoudData['a_position'][:, :, 0] > 0])])

            programSV3D.global_position(sv3D.lat, sv3D.lon, sv3D.yaw)
            if index == 0:
                ecef_offs = np.array(sv3D.ecef)
                ecef = np.array((0, 0, 0))
                programSV3D.offset_position(ecef)
                matrix_offs = programSV3D.u_model
                pano_offs = sv3DRegion.fileMeta['info2ID'][gps]

                #groundPoint = groundPoint_cur
            else:
                ecef = np.array(sv3D.ecef) - ecef_offs
                programSV3D.offset_position(ecef)

                #groundPoint = np.concatenate((groundPoint, groundPoint_cur), axis=0)

            if needMatchInfo3d:
                programSV3D.u_model = np.dot(programSV3D.u_model, np.linalg.inv(matrix_offs))
                programSV3D.info_3d_offs()

            sv3D.matrix_offs = programSV3D.u_model
            #gpyWindow.add_program(programSV3D)

            vec3 = np.reshape(sv3D.ptCLoudData['a_position'], (256 * 512, 3))
            vec4 = np.hstack([vec3, np.ones((len(vec3), 1))])
            vec4_mul = np.dot(vec4, sv3D.matrix_offs)
            vec4_out = np.reshape(vec4_mul[:, 0:3], (256, 512, 3))
            #groundPoint_cur = (sv3D.ptCLoudData[vec4_out[:, :, 2] < -2])
            groundPoint_cur['a_position'] = (vec4_out[(sv3D.ptCLoudData['a_position'][:, :, 2] > 2) * (sv3D.ptCLoudData['a_position'][:, :, 0] > 0)])

            if index == 0:
                matrix_offs_whole = sv3D.matrix_offs
                groundPoint = groundPoint_cur
            else:
                groundPoint = np.concatenate((groundPoint, groundPoint_cur), axis=0)

            index += 1

tri = np.array(triangle.delaunay(groundPoint['a_position'][:, 0:2]), dtype=np.uint32)
programGround = glumpy_setting.ProgramPlane(data=groundPoint, name=str(index), face=tri)
#programGround.u_model = matrix_offs_whole
gpyWindow.add_program(programGround)

programAxis = glumpy_setting.ProgramAxis(line_length=5)
gpyWindow.add_program(programAxis)

gpyWindow.run()

