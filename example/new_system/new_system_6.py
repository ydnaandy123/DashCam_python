#!/usr/bin/python3
# ==============================================================
# New system complete!!
# ==============================================================
import numpy as np
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import google_parse
import glumpy_setting

ID = '000034_info3d'
#ID = 'test_single'
gpyWindow = glumpy_setting.GpyWindow()
sv3DRegion = google_parse.StreetView3DRegion(ID)

index, ecef_offs = 0, np.zeros(3)
for sv3D in sv3DRegion.sv3D_list:
    programSV3D = glumpy_setting.ProgramSV3D(data=sv3D.ptCLoudData, name=str(index), point_size=1)
    programSV3D.global_position(sv3D.lat, sv3D.lon, sv3D.yaw)
    if index == 0:
        ecef_offs = np.array(sv3D.ecef)
        ecef = np.array((0, 0, 0))
    else:
        ecef = np.array(sv3D.ecef) - ecef_offs
    programSV3D.offset_position(ecef)
    gpyWindow.add_program(programSV3D)
    index += 1


programAxis = glumpy_setting.ProgramAxis(line_length=5)
gpyWindow.add_program(programAxis)

gpyWindow.run()