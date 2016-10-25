#!/usr/bin/python3
import numpy as np
import os
import json
#
import base_process

storePath = '/home/andy/src/DashCam/json/newSystem_deep_match'


class SFM3DRegion:
    def __init__(self, region_id):
        self.regionId = region_id
        self.dataDir = os.path.join(storePath, self.regionId, 'ransac_3D_result.json')
        with open(self.dataDir) as data_file:
            self.ransacResult = json.load(data_file)
            data_file.close()
        self.ptcloudData = parse_point_cloud_sfm(self.ransacResult)
        matrix = np.array(self.ransacResult['transformation'], dtype=np.float32)
        matrix = np.append(matrix, [0.0, 0.0, 0.0, 1.0])
        self.matrix = np.reshape(matrix, (4, 4))
        print(self.matrix)
        self.matrix = np.transpose(self.matrix)
        print(self.matrix)

        #self.ptcloudData['a_position'] = base_process.sv3d_apply_m4(data=self.ptcloudData['a_position'],
        #                                                    m4=self.matrix)


def sv3d_apply_m4(data, m4):
    vec4 = np.hstack([data, np.ones((len(data), 1))])
    vec4_mul = np.dot(vec4, m4)
    return vec4_mul[:, 0:3]


def parse_point_cloud_sfm(ransac_result):
    points_sfm = ransac_result['points']
    data_len = len(points_sfm.keys())
    data = np.zeros(data_len, dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
    cur = 0
    for key, value in points_sfm.items():
        data[cur]['a_position'] = value['coordinates']
        data[cur]['a_color'] = value['color']
        cur += 1
    data['a_color'] /= 255.0
    return data
