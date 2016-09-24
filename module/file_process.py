#!/usr/bin/python2
import json

class DashCamFileProcessor:
    def __init__(self):
        #self.loadList50()
        pass    
    
    # Load the seleced 50 files into self.list50
    def loadList50(self):
        with open('/home/andy/src/DashCam/json/namelist_50.json') as data_file:
            namelist50 = json.load(data_file)
            list50 = []
            for index, namelist in enumerate(namelist50):
                list50.append(namelist.split(','))
            self.list50 = list50
            data_file.close() 

    # Open the indicated file and parse it
    # return the non-repeat (lat, lon) pathPoint
    def getPath_info3d(self, fileID = None, fileIndex = None):
        if fileID != None:
            print('fileID: ' + fileID)
            with open('/home/andy/src/DashCam/json/newSystem_deep_match/' + fileID + '/info_3d.json') as data_file:
                info_3d = json.load(data_file)
                data_file.close()
        elif fileIndex != None:
            print('fileIndex: ' + fileIndex)
        pathPoint_set_info3d = set()
        for img, latlon in info_3d.items():
                for latlon_element in latlon.keys():		
                    if latlon_element not in pathPoint_set_info3d:	
                        pathPoint_set_info3d.add(latlon_element)  
        return pathPoint_set_info3d             