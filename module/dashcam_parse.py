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
        self.ptcloudData = parse_point_cloud_sfm(self.ransacResult['points'])
        self.trajectoryData = parse_trajectory(self.ransacResult['trajectory'])
        self.trajectoryJSON = self.ransacResult['trajectory']
        self.matrix = parse_matrix(self.ransacResult['transformation'])


def parse_point_cloud_sfm(points_sfm):
    data_len = len(points_sfm.keys())
    data = np.zeros(data_len, dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
    cur = 0
    for key, value in points_sfm.items():
        data[cur]['a_position'] = value['coordinates']
        data[cur]['a_color'] = value['color']
        cur += 1
    data['a_color'] /= 255.0
    return data


def parse_trajectory(trajectory):
    data_len = len(trajectory.keys())
    data = np.zeros(data_len, dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
    cur = 0
    for key, value in sorted(trajectory.items()):
        #print(key)
        data[cur]['a_position'] = value
        data[cur]['a_color'] = [1, 1, cur]
        cur += 1
    data['a_color'][:, 2] /= cur
    return data

def parse_matrix(transformation):
    matrix = np.array(transformation, dtype=np.float32)
    matrix = np.append(matrix, [0.0, 0.0, 0.0, 1.0])
    matrix = np.reshape(matrix, (4, 4))
    matrix = np.transpose(matrix)
    return matrix

