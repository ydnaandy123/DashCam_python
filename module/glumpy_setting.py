#!/usr/bin/python3
import numpy as np
import cv2
from glumpy import app, gloo, gl, glm
from glumpy.transforms import Rotate
from glumpy.ext import png

class Program:
    deg_x, deg_y, size, zoom, radius = 0, 0, 1, -200, 200
    datas, programsdatas = [], []

    def __init__(self, _vertex, _fragment):
        self.program = gloo.Program(_vertex, _fragment)
        self.program_axis = gloo.Program(vertex, _fragment)

        position = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        color = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])

        data = np.zeros((len(position)), dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])
        data['a_position'] = position * 100 
        data['a_color'] = color        

        data = data.view(gloo.VertexBuffer)
        self.program_axis.bind(data)

    def addPoint(self, data):
        data = data.view(gloo.VertexBuffer)
        self.datas.append(data)
        self.program.bind(data)


    def widowSetting(self):
        window = app.Window(1024,1024, color=(0,0,0,1))
        framebuffer = np.zeros((window.height, window.width * 3), dtype=np.uint8)
        @window.event
        def on_draw(dt):
            window.clear()
            # Filled program_cube
            self.program_axis.draw(gl.GL_LINES)
            self.program.draw(gl.GL_POINTS)
            # Make program_cube rotate
            self.program['u_model'] = matrix_model()  

        def matrix_model():
            model = np.eye(4, dtype=np.float32)
            glm.scale(model, self.size, self.size, self.size)
            glm.rotate(model, self.deg_y, 1, 0, 0)
            glm.rotate(model, self.deg_x, 0, 1, 0)
            #glm.translate(model, -self.deg_x/100, -self.deg_y/100, 0)
            #model[3,3] = 1
            return model      
        @window.event
        def on_resize(width, height):
            ratio = width / float(height)
            self.program['u_projection'] = glm.perspective(45.0, ratio, 0.001, 10000.0)

        @window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)   

        @window.event
        def on_mouse_scroll(x, y, dx, dy):
            self.size += dy*0.1 
            if self.size < 0:
                self.size = 0.1
            #self.zoom += dy*1
            #self.program['u_view'] = glm.translation(0, 0, self.zoom)

        @window.event
        def on_mouse_drag(x, y, dx, dy, button): 
            self.deg_y += dy # degrees
            self.deg_x += dx # degrees  

        @window.event
        def on_key_press(symbol, modifiers):
            if symbol == 88: # x
                self.view_orth_vector = np.array([self.radius, 0, 0])
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1], self.view_orth_vector[2])               
            elif symbol == 89: # y
                self.view_orth_vector = np.array([0, self.radius, 0])
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1], self.view_orth_vector[2])
            elif symbol == 90: # z
                self.view_orth_vector = np.array([0, 0, self.radius])
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1], self.view_orth_vector[2])
            elif symbol == 70: # f
                self.view_orth_vector = -self.view_orth_vector
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1], self.view_orth_vector[2])
            elif symbol == 80: # p
                gl.glReadPixels(0, 0, window.width, window.height,
                                gl.GL_RGB, gl.GL_UNSIGNED_BYTE, framebuffer)
                png.from_array(framebuffer, 'RGB').save('screenshot.png')                 
                   
            print('Key pressed (symbol=%s, modifiers=%s)'% (symbol,modifiers))  
            #self.program['color_sel'] = 1 - self.program['color_sel']


        self.program['color'] = (1,0,0,1)
        self.program['color_sel'] = 1
        self.program['u_model'] = np.eye(4, dtype=np.float32)
        self.program['u_view'] = glm.translation(0, 0, -50)
        self.program['a_pointSize'] = 5

        app.run()



'''
data_ptCloud_streetView3D = sv3D.data_ptCLoud
program_ptCloud = gloo.Program(shader.vertex, shader.fragment) 
data_ptCloud_streetView3D = data_ptCloud_streetView3D.view(gloo.VertexBuffer);
program_ptCloud.bind(data_ptCloud_streetView3D)
#program_ptCloud['a_position'] = data_ptCloud_streetView3D['a_position']
#program_ptCloud['a_color'] = data_ptCloud_streetView3D['a_color']

program_ptCloud['color'] = 1,0,0,1
program_ptCloud['color_sel'] = 1
program_ptCloud['u_model'] = np.eye(4, dtype=np.float32)
program_ptCloud['u_view'] = glm.translation(0, 0, -200)
self.deg_x, self.deg_y, self.size, self.zoom = 0, 0, 1, -200

widowSetting();
'''

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
    //gl_PointSize = a_pointSize;
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