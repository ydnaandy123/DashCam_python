#!/usr/bin/python3
import numpy as np
import cv2
from glumpy import app, gloo, gl, glm
from glumpy.transforms import Rotate
from glumpy.ext import png
import shader

def widowSetting():
    window = app.Window(1024,1024, color=(0,0,0,1))
    framebuffer = np.zeros((window.height, window.width * 3), dtype=np.uint8)
    @window.event
    def on_draw(dt):
        window.clear()
        # Filled program_cube
        program_ptCloud.draw(gl.GL_POINTS)

        # Make program_cube rotate
        program_ptCloud['u_model'] = matrix_model()  

    def matrix_model():
        model = np.eye(4, dtype=np.float32)
        #model *= size
        glm.rotate(model, theta, 1, 0, 0)
        glm.rotate(model, -phi, 0, 1, 0)
        #glm.translate(model, -phi/100, -theta/100, 0)
        glm.scale(model, size)
        #model[3,3] = 1
        return model      
    @window.event
    def on_resize(width, height):
        ratio = width / float(height)
        program_ptCloud['u_projection'] = glm.perspective(45.0, ratio, 0.001, 10000.0)

    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)   

    @window.event
    def on_mouse_scroll(x, y, dx, dy):
        global zoom, size
        size += dy*0.1 
        if size < 0:
            size = 0.1
        #zoom += dy*1
        #program_ptCloud['u_view'] = glm.translation(0, 0, zoom)

    @window.event
    def on_mouse_drag(x, y, dx, dy, button): 
        global theta, phi
        theta += dy # degrees
        phi -= dx # degrees      

    @window.event
    def on_key_press(symbol, modifiers):
        gl.glReadPixels(0, 0, window.width, window.height,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, framebuffer)
        png.from_array(framebuffer, 'RGB').save('screenshot.png')        
        #print('Key pressed (symbol=%s, modifiers=%s)'% (symbol,modifiers))  
        program_ptCloud['color_sel'] = 1 - program_ptCloud['color_sel']

    app.run()




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
phi, theta, size, zoom = 0, 0, 1, -200

widowSetting();
