#!/usr/bin/python2
import json
import os

class DashCamFileProcessor:
    def __init__(self):
        #self.loadList50()
        pass    

    def loadList50(self):
        with open('src/json/dashcam/namelist_50.json') as data_file:    
            namelist50 = json.load(data_file)
            list50 = []
            for index, namelist in enumerate(namelist50):
                list50.append(namelist.split(','))
            self.list50 = list50
            data_file.close() 

    def getPath_info3d(self, fileID = None, fileIndex = None):
        if fileID != None:
            print fileID
            with open('src/json/dashcam/deep_match/' + fileID + '/info_3d.json') as data_file:    
                info_3d = json.load(data_file)
                data_file.close()
        pathPoint_set_info3d = set()
        for img, latlon in info_3d.items():
                for latlon_element in latlon.keys():		
                    if latlon_element not in pathPoint_set_info3d:	
                        pathPoint_set_info3d.add(latlon_element)  
        return pathPoint_set_info3d             