#!/usr/bin/python3
# ==============================================================
# Showing how to use StreetView3DRegion to parse a hole region
# ==============================================================
import numpy as np
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import google_parse
import glumpy_setting

ID = '000034_info3d'
sv3DRegion = google_parse.StreetView3DRegion(ID)

gpyWindow = glumpy_setting.GpyWindow()
programAxis = glumpy_setting.ProgramAxis(line_length=2)
gpyWindow.add_program(programAxis)
programSV3D = glumpy_setting.ProgramSV3D(sv3DRegion.sv3D_list[0].ptCLoudData)
gpyWindow.add_program(programSV3D)
gpyWindow.run()