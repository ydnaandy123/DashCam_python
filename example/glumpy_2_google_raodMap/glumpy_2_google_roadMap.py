#!/usr/bin/python3
import numpy as np
import cv2
from glumpy import app, gloo, gl, glm
from glumpy.transforms import Rotate
import os
import json
import shader
import sys
sys.path.append('module')	# use the module under Module 
import google_parse
import base_process
import timeit
ID = '000731';
fileDir = 'src/panometa/' + ID 
ImgDir = 'src/panorama/' + ID 
panoNum = len(os.listdir(fileDir))
data = np.zeros((panoNum), dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])

for idx, fileName in enumerate(os.listdir(fileDir)):
    with open(fileDir + '/' + fileName) as data_file:    
        panoMeta = json.load(data_file)
        panorama =  cv2.imread(ImgDir + '/' + fileName.split('.')[0] + '.jpg', cv2.IMREAD_COLOR)
        data[idx]['a_position'] = base_process.GPS2GL(panoMeta['Lat'], panoMeta['Lon'])
        

data['a_color'] = [0,1,0]   
data['a_position'] -= data[0]['a_position']
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
        glm.translate(model, tx, ty, 0)
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
        if(size + dy*0.1 > 0.1):
            size += dy*0.1
        else:
            size = 0.1
        #zoom += dy*1
        #program_ptCloud['u_view'] = glm.translation(0, 0, zoom)

    @window.event
    def on_mouse_drag(x, y, dx, dy, button):
        if button == 2: 
            global theta, phi, tx, ty
            theta += dy # degrees
            phi -= dx # degrees
        elif button == 8:
            tx += dx
            ty -= dy

    @window.event
    def on_key_press(symbol, modifiers):
        gl.glReadPixels(0, 0, window.width, window.height,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, framebuffer)
        png.from_array(framebuffer, 'RGB').save('screenshot.png')        
        #print('Key pressed (symbol=%s, modifiers=%s)'% (symbol,modifiers))  
        program_ptCloud['color_sel'] = 1 - program_ptCloud['color_sel']

    app.run()


tic=timeit.default_timer()
#sv3D.CreatePtCloud2(sphericalRay)
toc=timeit.default_timer()
#data_ptCloud_streetView3D = sv3D.data_ptCLoud
data_ptCloud_streetView3D = data
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
tx, ty = 0, 0

widowSetting();