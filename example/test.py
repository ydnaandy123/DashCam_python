#!/user/bin/python2
import json

with open('src/json/dashcam/namelist_50.json') as data_file:    
    namelist_50 = json.load(data_file)
    for namelist in namelist_50:
        [fileIndex, fileID] = namelist.split(',')
        #print fileID
    data_file.close()