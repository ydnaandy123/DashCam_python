#!/usr/bin/python3
# ==============================================================
# Showing how to parse the data downloaded from google
# data is stored in BFS or info_3d
# ==============================================================
import os
import timeit
import numpy as np
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
            #print(panoId)
            #if panoId != 'ny852t6BX7kEExOQ5APm6A':
            #    continue
            #print(panoId)
            panoIdDir = os.path.join(dataDir, panoId)
            panorama = scipy.misc.imread(panoIdDir + '.jpg').astype(np.float)
            with open(panoIdDir + '.json') as data_file:
                panoMeta = json.load(data_file)
                data_file.close()
            break
        meta_file.close()
else:
    # info_3d
    panoMeta, panorama = [], []
    pass

sv3D = google_parse.StreetView3D(panoMeta, panorama)
# All the google depth maps seem to be store
# as this size(256, 512), at least what I've seen
sphericalRay = google_parse.create_spherical_ray(256, 512)
tic = timeit.default_timer()
sv3D.create_ptcloud_old_version(sphericalRay)
toc = timeit.default_timer()
print('The old way takes %.8f' % (toc-tic))
# Visualize
#sv3D.show_depth()

tic = timeit.default_timer()
sv3D.create_ptcloud(sphericalRay)
toc = timeit.default_timer()
print('The new way takes %.8f' % (toc-tic))
# Visualize
#sv3D.show_depth()

vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
uniform float  a_pointSize;
attribute vec3 a_position;      // Vertex position
attribute vec3 a_color;
varying vec3 v_color;
void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    //gl_LineWidth = a_pointSize;
    gl_PointSize = a_pointSize;
    v_color = a_color;
}
"""

fragment = """
varying vec3 v_color;
uniform vec4 color;
uniform int color_sel;
void main()
{
    if(color_sel == 1){
        gl_FragColor = vec4(v_color, 1.0);
    }
    else{
        gl_FragColor = color;
    }
}
"""
streetView3D = glumpy_setting.Program(vertex, fragment)
streetView3D.addPoint(sv3D.ptCLoudData)
streetView3D.widowSetting()
