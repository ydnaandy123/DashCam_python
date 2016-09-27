#!/usr/bin/python2
import json


class DashCamFileProcessor:
    def __init__(self):
        self.list50 = []

        self.load_list_50()

    """
    # Load the selected 50 files into self.list50
    """
    def load_list_50(self):
        with open('/home/andy/src/DashCam/json/namelist_50.json') as data_file:
            namelist50 = json.load(data_file)
            list50 = []
            for index, namelist in enumerate(namelist50):
                list50.append(namelist.split(','))
            self.list50 = list50
            data_file.close() 
    """
    # Open the indicated file and parse it
    # return the non-repeat (lat, lon) pathPoint
    """
    @staticmethod
    def get_path_info3d(file_id = None, file_index = None):
        if file_id is not None:
            print('fileID: ' + file_id)
            with open('/home/andy/src/DashCam/json/newSystem_deep_match/' + file_id + '/info_3d.json') as data_file:
                info_3d = json.load(data_file)
                data_file.close()
        elif file_index is not None:
            info_3d = None
            print('fileIndex: ' + file_index)
        path_point_set_info3d = set()
        for img, gps in info_3d.items():
                for gps_element in gps.keys():
                    if gps_element not in path_point_set_info3d:
                        path_point_set_info3d.add(gps_element)
        return path_point_set_info3d