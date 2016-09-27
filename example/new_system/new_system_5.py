#!/usr/bin/python3
# ==============================================================
# Showing new system (ECEF coordinate)
# ==============================================================
import numpy as np
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import google_parse
import glumpy_setting

#ID = '000034_info3d'
ID = 'test_single'
sv3DRegion = google_parse.StreetView3DRegion(ID)

gpyWindow = glumpy_setting.GpyWindow()
programAxis = glumpy_setting.ProgramAxis(line_length=2)
gpyWindow.add_program(programAxis)

sv3D = sv3DRegion.sv3D_list[0]
yaw = sv3D.panoMeta['ProjectionPanoYawDeg']
lat = sv3D.panoMeta['Lat']
lon = sv3D.panoMeta['Lon']
print(yaw, sv3D.ecef)
programSV3D = glumpy_setting.ProgramSV3D(sv3D.ptCLoudData)
programSV3D.global_position(float(lat), float(lon), float(yaw))
gpyWindow.add_program(programSV3D)
gpyWindow.run()