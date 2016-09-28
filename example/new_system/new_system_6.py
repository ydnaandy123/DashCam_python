#!/usr/bin/python3
# ==============================================================
# New system complete!!
# ==============================================================
import numpy as np
import json
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting

# Create dashCamFileProcess and load 50 top Dashcam
fileIndex = 16
dashCamFileProcess = file_process.DashCamFileProcessor()
fileID = str(dashCamFileProcess.list50[fileIndex][1])
print(fileID, fileIndex)
fileID += '_info3d'

isMatchInfo3d = False

gpyWindow = glumpy_setting.GpyWindow()
sv3DRegion = google_parse.StreetView3DRegion(fileID)

index, ecef_offs, matrix_offs = 0, np.zeros(3), np.eye(4, dtype=np.float32)
for sv3D_key in sv3DRegion.sv3D_Dict:
    sv3D = sv3DRegion.sv3D_Dict[sv3D_key]
    programSV3D = glumpy_setting.ProgramSV3D(data=sv3D.ptCLoudData, name=str(index), point_size=1)
    programSV3D.global_position(sv3D.lat, sv3D.lon, sv3D.yaw)
    if index == 0:
        ecef_offs = np.array(sv3D.ecef)
        ecef = np.array((0, 0, 0))
        programSV3D.offset_position(ecef)
        matrix_offs = programSV3D.u_model
    else:
        ecef = np.array(sv3D.ecef) - ecef_offs
        programSV3D.offset_position(ecef)

    if isMatchInfo3d:
        programSV3D.u_model = np.dot(programSV3D.u_model, np.linalg.inv(matrix_offs))
        programSV3D.info_3d_offs()

    sv3D.matrix = programSV3D.u_model

    ## reset
    #vec3 = np.reshape(sv3D.ptCLoudData['a_position'], (256*512, 3))
    #vec4 = np.hstack([vec3, np.ones((len(vec3), 1))])
    #vec4_mul = np.dot(vec4, sv3D.matrix)
    #sv3D.ptCLoudData['a_position'] = np.reshape(vec4_mul[:, 0:3], (256, 512, 3))

    #programSV3D = glumpy_setting.ProgramSV3D(data=sv3D.ptCLoudData, name=str(index), point_size=1)
    #programSV3D.u_model = np.eye(4, dtype=np.float32)
    ## reset

    gpyWindow.add_program(programSV3D)
    index += 1


programAxis = glumpy_setting.ProgramAxis(line_length=5)
gpyWindow.add_program(programAxis)

gpyWindow.run()

