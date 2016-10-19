#!/usr/bin/python3
# ==============================================================
# Step.1
# Transform the point in SFM to Google (SV3D)
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
needMatchInfo3d = True
dashCamFileProcess = file_process.DashCamFileProcessor()
gpyWindow = glumpy_setting.GpyWindow()
"""
For Visual
"""
data_info, sleIndex = [], 9
for fileIndex in range(sleIndex,sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)

    with open('/home/andy/src/DashCam/json/newSystem_deep_match/' + fileID + '/info_3d.json') as data_file:
        info_3d = json.load(data_file)
        data_file.close()
    fileID += '_info3d'

    sv3DRegion = google_parse.StreetView3DRegion(fileID)
    sv3DRegion.init_region(anchor=None)

    anchor_matrix_whole = sv3DRegion.anchorMatrix
    anchorId, anchorLat, anchorLon = sv3DRegion.anchorId, sv3DRegion.anchorLat, sv3DRegion.anchorLon

    index = 0

    for img, gps in info_3d.items():
        for gps, indices in info_3d[img].items():
            print(gps, sv3DRegion.fileMeta['info2ID'][gps], sv3DRegion.fileMeta['id2GPS'][sv3DRegion.fileMeta['info2ID'][gps]])
            sv3D = sv3DRegion.sv3D_Dict[sv3DRegion.fileMeta['info2ID'][gps]]
            sv3D.apply_global_adjustment()
            sv3D.apply_local_adjustment()

            programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=sv3D.ptCLoudData, name=None, point_size=1,
                                                                 anchor_matrix=anchor_matrix_whole)
            programSV3DRegion.apply_anchor()
            gpyWindow.add_program(programSV3DRegion)

            # output
            vec4_mul = sv3D.ptCLoudData['a_position']
            for idx in range(len(indices)):
                sel = info_3d[img][gps][idx]
                info_3d[img][gps][idx] = []
                info_3d[img][gps][idx].append(float(vec4_mul[sel, 0]))
                info_3d[img][gps][idx].append(float(vec4_mul[sel, 1]))
                info_3d[img][gps][idx].append(float(vec4_mul[sel, 2]))
                #data_info.append(vec4_mul[sel, 0:3])
                data_info.append([float(vec4_mul[sel, 0]), float(vec4_mul[sel, 1]), float(vec4_mul[sel, 2])])

            # info_3d match
            match_info = np.zeros(len(data_info), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
            match_info['a_position'] = data_info
            match_info['a_color'] = [1, 0, 0]
            programMatch = glumpy_setting.ProgramSV3D(data=match_info, name=str(index), point_size=7)
            gpyWindow.add_program(programMatch)

            index += 1

    """
    For output
    """
    with open('/home/andy/src/DashCam/json/info_2_gps/' + fileID + '_2_gps.json', 'w') as outfile:
        json.dump(info_3d, outfile)
        outfile.close()

    with open('/home/andy/src/DashCam/json/info_2_gps_offs/' + fileID + '_pano_offs.json', 'w') as outfile:
        file_meta = {'anchorId': anchorId, 'anchorLat': anchorLat, 'anchorLon': anchorLon}
        json.dump(file_meta, outfile)


programAxis = glumpy_setting.ProgramAxis(line_length=5)
gpyWindow.add_program(programAxis)

gpyWindow.run()

