#!/usr/bin/python3
# ==============================================================
# New system complete!!
# ==============================================================
import numpy as np
import json
import sys
import simplejson

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_parse
import glumpy_setting

# Create dashCamFileProcess and load 50 top Dashcam
isMatchInfo3d = True
dashCamFileProcess = file_process.DashCamFileProcessor()
#gpyWindow = glumpy_setting.GpyWindow()

for fileIndex in range(50):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)

    with open('/home/andy/src/DashCam/json/newSystem_deep_match/' + fileID + '/info_3d.json') as data_file:
        info_3d = json.load(data_file)
        data_file.close()
    fileID += '_info3d'

    sv3DRegion = google_parse.StreetView3DRegion(fileID)

    index, ecef_offs, matrix_offs = 0, np.zeros(3), np.eye(4, dtype=np.float32)

    for img, gps in info_3d.items():
        for gps, indices in info_3d[img].items():
            print(gps, sv3DRegion.fileMeta['info2ID'][gps], sv3DRegion.fileMeta['id2GPS'][sv3DRegion.fileMeta['info2ID'][gps]])
            sv3D = sv3DRegion.sv3D_Dict[sv3DRegion.fileMeta['info2ID'][gps]]
            #for idx in range(len(indices)):
            #    info_3d[img][gps][idx] = (0, 0, 0)
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
                #color_vec3[sel] = [1,0,0]

            vec4_out = np.reshape(vec4_mul[:, 0:3], (256, 512, 3))
            sv3D.ptCLoudData['a_position'] = vec4_out
            #sv3D.ptCLoudData['a_color'] = np.reshape(color_vec3, (256, 512, 3))

            programSV3D = glumpy_setting.ProgramSV3D(data=sv3D.ptCLoudData, name=str(index), point_size=1)
            programSV3D.u_model = np.eye(4, dtype=np.float32)
            ##reset

            #gpyWindow.add_program(programSV3D)
            index += 1

    with open('/home/andy/src/DashCam/json/info_2_gps/' + fileID + '_2_gps.json', 'w') as outfile:
        simplejson.dump(info_3d, outfile)
        outfile.close()


    with open('/home/andy/src/DashCam/json/info_2_gps_offs/' + fileID + '_pano_offs.json', 'w') as outfile:
        file_meta = {'pano_offs': pano_offs}
        json.dump(file_meta, outfile)


#programAxis = glumpy_setting.ProgramAxis(line_length=5)
#gpyWindow.add_program(programAxis)

#gpyWindow.run()

