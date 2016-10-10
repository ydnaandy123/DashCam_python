import numpy as np
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')
import glumpy_setting


n_point = 100
sx, sy = 10, 10
ground = np.ones((n_point, n_point)) * -2
x = np.linspace(-sx, sx, n_point)
y = np.linspace(-sy, sy, n_point)
xv, yv = np.meshgrid(x, y)

data = np.zeros((n_point, n_point), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])

data['a_position'] = np.dstack((xv, yv, ground))
data['a_color'] = (1, 0, 0)


gpyWindow = glumpy_setting.GpyWindow()

programSV3DRegion = glumpy_setting.ProgramSV3DRegion(data=data, name=None, point_size=1)
gpyWindow.add_program(programSV3DRegion)

programAxis = glumpy_setting.ProgramAxis(line_length=5)
gpyWindow.add_program(programAxis)

gpyWindow.run()
