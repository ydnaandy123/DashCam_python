#!/usr/bin/python3
# ==============================================================
# Showing how to use glumpy_setting to visualize SV3D
# ==============================================================
import numpy as np
import os
import json
import scipy.misc
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import google_parse
import glumpy_setting

ID = '000067'
dataDir = os.path.join('/home/andy/src/Google/panometa', ID)
metaDir = os.path.join(dataDir, "fileMeta.json")

if os.path.exists(metaDir):
    # BFS
    with open(metaDir) as meta_file:
        fileMeta = json.load(meta_file)
        for panoId in fileMeta['id2GPS']:
            print(panoId)
            if panoId != 'ny852t6BX7kEExOQ5APm6A':
                continue
            print(panoId, fileMeta['id2GPS'][panoId])
            panoIdDir = os.path.join(dataDir, panoId)
            panorama = scipy.misc.imread(panoIdDir + '.jpg').astype(np.float)
            with open(panoIdDir + '.json') as data_file:
                panoMeta = json.load(data_file)
                data_file.close()
        meta_file.close()
else:
    # info_3d
    panoMeta, panorama = [], []
    pass

sv3D = google_parse.StreetView3D(panoMeta, panorama)
# All the google depth maps seem to be store
# as this size(256, 512), at least what I've seen
sphericalRay = google_parse.create_spherical_ray(256, 512)
sv3D.create_ptcloud(sphericalRay)
# Visualize
#sv3D.show_depth()

gpyWindow = glumpy_setting.GpyWindow()
programAxis = glumpy_setting.ProgramAxis(line_length=2)
gpyWindow.add_program(programAxis)

m4 = np.array([[-0.52218324,  0.        , -0.85283333,  0.        ],
 [-0.36136094,  0.90579408,  0.22125851,  0.        ],
 [ 0.7724914,   0.42371812, -0.47299048,  0.        ],
 [ 0.,          0.,          0.,          1.        ]])

p = np.dot(np.reshape(sv3D.ptCLoudData['a_position'], (256*512, 3)), m4[0:3, 0:3])
sv3D.ptCLoudData['a_position'] = np.reshape(p, (256, 512, 3))
programSV3D = glumpy_setting.ProgramSV3D(sv3D.ptCLoudData)
gpyWindow.add_program(programSV3D)
gpyWindow.run()