#!/usr/bin/python3
import numpy as np
import sys
sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')	# use the module under Module
import glumpy_setting
import google_parse

fileID = '000067'
streetView3DRegion = google_parse.StreetView3DRegion(fileID)
topology = streetView3DRegion.createTopoloy()

position = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
color = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])

data = np.zeros((len(position)), dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])
data['a_position'] = position * 100 
data['a_color'] = color

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

topology['a_position'] -= topology['a_position'][0]
topology['a_position'][:, 2] = -topology['a_position'][:, 2]
streetView3D = glumpy_setting.Program(vertex, fragment)
streetView3D.addPoint(np.append(topology, data))
#streetView3D.addPoint(data)
streetView3D.widowSetting()