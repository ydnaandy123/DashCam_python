#!/usr/bin/python3
# ==============================================================
# Delaunay!!
# ==============================================================
import numpy as np
import json
import sys
from glumpy import app, gl, gloo
import triangle

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting

# Create dashCamFileProcess and load 50 top Dashcam
isMatchInfo3d = True
dashCamFileProcess = file_process.DashCamFileProcessor()
gpyWindow = glumpy_setting.GpyWindow()

fileIndex = 1
fileID = str(dashCamFileProcess.list50[fileIndex][1])
print(fileID, fileIndex)

with open('/home/andy/src/DashCam/json/newSystem_deep_match/' + fileID + '/info_3d.json') as data_file:
    info_3d = json.load(data_file)
    data_file.close()
fileID += '_info3d'

sv3DRegion = google_parse.StreetView3DRegion(fileID)

index, ecef_offs, matrix_offs = 0, np.zeros(3), np.eye(4, dtype=np.float32)

match = []
match_c = []
for img, gps in sorted(info_3d.items()):
    for gps, indices in info_3d[img].items():
        print(gps, sv3DRegion.fileMeta['info2ID'][gps], sv3DRegion.fileMeta['id2GPS'][sv3DRegion.fileMeta['info2ID'][gps]])
        sv3D = sv3DRegion.sv3D_Dict[sv3DRegion.fileMeta['info2ID'][gps]]
        programSV3D = glumpy_setting.ProgramSV3D(data=sv3D.ptCLoudData, name=str(index), point_size=1)
        programSV3D.global_position(sv3D.lat, sv3D.lon, sv3D.yaw)
        if index == 0:
            ecef_offs = np.array(sv3D.ecef)
            ecef = np.array((0, 0, 0))
            programSV3D.offset_position(ecef)
            matrix_offs = programSV3D.u_model
            pano_offs = sv3DRegion.fileMeta['info2ID'][gps]
        else:
            ecef = np.array(sv3D.ecef) - ecef_offs
            programSV3D.offset_position(ecef)

        if isMatchInfo3d:
            programSV3D.u_model = np.dot(programSV3D.u_model, np.linalg.inv(matrix_offs))
            programSV3D.info_3d_offs()

        sv3D.matrix_offs = programSV3D.u_model

        ## reset
        vec3 = np.reshape(sv3D.ptCLoudData['a_position'], (256 * 512, 3))
        vec4 = np.hstack([vec3, np.ones((len(vec3), 1))])
        vec4_mul = np.dot(vec4, sv3D.matrix_offs)
        #color_vec3 = np.reshape(sv3D.ptCLoudData['a_color'], (256 * 512, 3))
        '''
        output
        '''
        for idx in range(len(indices)):
            sel = info_3d[img][gps][idx]
            info_3d[img][gps][idx] = []
            info_3d[img][gps][idx].append(vec4_mul[sel, 0])
            info_3d[img][gps][idx].append(vec4_mul[sel, 1])
            info_3d[img][gps][idx].append(vec4_mul[sel, 2])
            #if vec4_mul[sel, 2] < 1:
                #match.append([vec4_mul[sel, 0], vec4_mul[sel, 1], 0])
                #match_c
            #color_vec3[sel] = [1,0,0]

        vec4_out = np.reshape(vec4_mul[:, 0:3], (256, 512, 3))
        sv3D.ptCLoudData['a_position'] = vec4_out
        #sv3D.ptCLoudData['a_color'] = np.reshape(color_vec3, (256, 512, 3))

        programSV3D = glumpy_setting.ProgramSV3D(data=sv3D.ptCLoudData, name=str(index), point_size=1)
        programSV3D.u_model = np.eye(4, dtype=np.float32)
        ##reset

        #yo = np.where(sv3D.ptCLoudData['a_position'][:, :, 2] < 1)
        match = sv3D.ptCLoudData[sv3D.ptCLoudData['a_position'][:, :, 2] < -1]
        match['a_position'][:, 2] = -2
        #match = sv3D.ptCLoudData['a_color'][sv3D.ptCLoudData['a_position'][:, :, 2] < 1]
        #gpyWindow.add_program(programSV3D)
        index += 1
        break

#programAxis = glumpy_setting.ProgramAxis(line_length=5)
#gpyWindow.add_program(programAxis)


#match_data = np.zeros(len(match), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
#match_data['a_position'] = match
#match_data['a_color'] = [1,0,0]

tri = np.array(triangle.delaunay(match['a_position'][:, 0:2]), dtype=np.uint32)
tri = tri.view(gloo.IndexBuffer)

matchProgram = glumpy_setting.ProgramPlane(data=match, name=str(index), face=tri)
gpyWindow.add_program(matchProgram)

gpyWindow.run()

