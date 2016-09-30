# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
from plyfile import PlyData, PlyElement
import numpy
from glumpy import glm
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process


vertex = numpy.array([(0, 0, 0),
                      (0, 1, 1),
                      (1, 0, 1),
                      (1, 1, 0)],
                     dtype=[('x', 'f4'), ('y', 'f4'),
                            ('z', 'f4')])

face = numpy.zeros((1), dtype=[('vertex_indices', 'i4', 3)])

el1 = PlyElement.describe(vertex, 'some_name')
el2 = PlyElement.describe(face, 'some_name2')
PlyData([el1, el2], text=True).write('some_ascii.ply')
'''

id = 'uSjqj9Lt256V8I7RckMykA'
pano = google_store.get_panorama_metadata(panoid=id)
print(pano.id)

'''