#!/usr/bin/python3
import sys
sys.path.append('module')	# use the module under Module
import glumpy_setting
import google_parse

fileID = 'test'
streetView3DRegion = google_parse.StreetView3DRegion(fileID)
topology = streetView3DRegion.createTopoloy()
#data = np.zeros((100), dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])

streetView3D = glumpy_setting.ProgramSV3D(topology)
streetView3D.widowSetting()
