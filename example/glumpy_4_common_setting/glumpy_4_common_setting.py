#!/usr/bin/python3
import sys
sys.path.append('module')	# use the module under Module
import glumpy_setting
import google_parse

fileID = 'test'
streetView3DRegion = google_parse.StreetView3DRegion(fileID)
topology = streetView3DRegion.createTopoloy()
import numpy as np
data = np.zeros((7), dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])

data['a_color'] = [[1, 1, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
data['a_position'] = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
print (topology)
topology['a_position']  = topology['a_position'] / 6371000.0 * 20
streetView3D = glumpy_setting.Program()
streetView3D.addPoint(data)
streetView3D.widowSetting()
