#!/usr/bin/python
import numpy as np
from glumpy import app, gloo, gl, glm
from glumpy.transforms import Rotate
from glumpy.ext import png
import json
import shader

ID = '000067';
with open('src/json/dashcam/deep_match/' + ID + '/pointCloud.json') as data_file:    
    pointCloud = json.load(data_file)

with open('src/json/dashcam/detection_box_result/' + ID + '/box_result_3d.json') as data_file:    
    box = json.load(data_file)
    XYZ = []
    for img in box:
        for car in box[img]:
            XYZ.append([box[img][car][0][0][0][0], box[img][car][0][0][1][0], box[img][car][0][0][2][0]])
    
    data_len = len(box)
    data = np.zeros(data_len, dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])
    data['a_position'] = np.asarray(XYZ, dtype = np.float32)
    data['a_color'] = [255,0,0]

def parsePointClout_SFM(pointCloud):
    points_SFM = pointCloud['points']
    data_len = len(points_SFM.keys())
    data = np.zeros(data_len, dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])
    cur = 0
    for key, value in points_SFM.items():
        data[cur]['a_position'] = value['coordinates']
        data[cur]['a_color'] = value['color']
        cur += 1   
    return data 

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
        model *= size
        glm.rotate(model, theta, 1, 0, 0)
        glm.rotate(model, phi, 0, 0, 1)
        model[3,3] = 1
        return model      
    @window.event
    def on_resize(width, height):
        ratio = width / float(height)
        program_ptCloud['u_projection'] = glm.perspective(45.0, ratio, 0.1, 1000.0)

    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)   

    @window.event
    def on_mouse_scroll(x, y, dx, dy):
        global zoom 
        zoom += dy*20
        program_ptCloud['u_view'] = glm.translation(0, 0, zoom)

    @window.event
    def on_mouse_drag(x, y, dx, dy, button): 
        global theta, phi
        theta += dy # degrees
        phi + dx # degrees      

    @window.event
    def on_key_press(symbol, modifiers):
        gl.glReadPixels(0, 0, window.width, window.height,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, framebuffer)
        png.from_array(framebuffer, 'RGB').save('screenshot.png')        
        #print('Key pressed (symbol=%s, modifiers=%s)'% (symbol,modifiers))  
        program_ptCloud['color_sel'] = 1 - program_ptCloud['color_sel']

    app.run()


data_ptCloud_SFM = parsePointClout_SFM(pointCloud)
data_ptCloud_SFM = np.append(data,data_ptCloud_SFM)
print (data_ptCloud_SFM)
program_ptCloud = gloo.Program(shader.vertex, shader.fragment) 

data_ptCloud_SFM = data_ptCloud_SFM.view(gloo.VertexBuffer);
program_ptCloud.bind(data_ptCloud_SFM)
#program_ptCloud['a_position'] = data_ptCloud_SFM['a_position']
#program_ptCloud['a_color'] = data_ptCloud_SFM['a_color']

program_ptCloud['color'] = 1,0,0,1
program_ptCloud['color_sel'] = 1
program_ptCloud['u_model'] = np.eye(4, dtype=np.float32)
program_ptCloud['u_view'] = glm.translation(0, 0, -200)
phi, theta, size, zoom = 40, 30, 1, -200

widowSetting();
