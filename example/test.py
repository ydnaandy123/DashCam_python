# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import triangle
from scipy.spatial import Delaunay
from glumpy import app, gl, gloo
import numpy as np
from glumpy.geometry import colorcube

position = np.array([(-0.8, -1.2), (-1.4, +0.5), (0, 1.2), (+1.4, +0.5), (+0.8, -1.2)])
color = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0]])

#tri = Delaunay(position)


triDict = {'vertices': position}
pp = triangle.triangulate(triDict)


tri = np.array(triangle.triangulate(triDict), dtype=np.uint32)
tri = tri.view(gloo.IndexBuffer)

vertices, faces, outline = colorcube()

vertex = """
attribute vec2 a_position;
attribute vec3 a_color;
varying vec3 v_color;
void main (void)
{
    gl_Position = vec4(0.65*a_position, 0.0, 1.0);
    gl_PointSize = 50;
    v_color = a_color;
}
"""

fragmentSimple = """
varying vec3 v_color;
void main()
{
    gl_FragColor = vec4(v_color, 1.0);
}
"""

window = app.Window(width=800, height=800)

@window.event
def on_draw(dt):
    window.clear()

    gl.glDisable(gl.GL_BLEND)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)

    quad.draw(gl.GL_TRIANGLE_STRIP, indices=tri)
    #quad.draw(gl.GL_TRIANGLE_STRIP, program_object.face)

quad = gloo.Program(vertex, fragmentSimple)
quad['a_position'] = position
quad['a_color'] = color
app.run()