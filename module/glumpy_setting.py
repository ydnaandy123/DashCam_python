#!/usr/bin/python3
import numpy as np
from glumpy import app, gloo, gl, glm
from glumpy.ext import png
#
import base_process

class ProgramAxis():
    def __init__(self, data=None, name='ProgramAxis', line_length=1, arrow_size=2, geometry_size=1):
        program = gloo.Program(vertexSimple, fragmentSimple)
        axis = np.array([[-10, 0, 0], [10, 0, 0],
                         [0, -10, 0], [0, 10, 0],
                         [0, 0, -10], [0, 0, 10]],
                        dtype=np.float32)
        axis, ends = axis * line_length, 10 * line_length
        arrow = []
        for i in range(3):
            arrow_one = []
            arrow_one.append([ends, arrow_size / 2, 0])
            arrow_one.append([ends + arrow_size, 0, 0])
            arrow_one.append([ends, -arrow_size / 2, 0])
            arrow_one.append([ends + arrow_size, 0, 0])
            arrow_one.append([ends, arrow_size / 2, 0])
            arrow_one.append([ends, -arrow_size / 2, 0])
            arrow_one = np.array(arrow_one, dtype=np.float32)
            if i == 1:
                x = np.copy(arrow_one[:, 0])
                arrow_one[:, 0] = arrow_one[:, 1]
                arrow_one[:, 1] = x
            elif i == 2:
                x = np.copy(arrow_one[:, 0])
                arrow_one[:, 0] = arrow_one[:, 2]
                arrow_one[:, 2] = x
            arrow.append(arrow_one)
        arrow = np.array(arrow, dtype=np.float32)
        geometry = np.append(axis, arrow)
        geometry *= geometry_size
        program['a_position'] = geometry
        program['a_color'] = np.array([[1, 0, 0], [1, 0, 0],
                                       [0, 1, 0], [0, 1, 0],
                                       [0, 0, 1], [0, 0, 1],
                                       [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                                       [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                                       [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],])

        self.data, self.name = data, name
        self.program = program
        self.draw_mode = gl.GL_LINES
        self.u_model, self.u_view, self.u_projection = np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), np.eye(
            4, dtype=np.float32)


class ProgramSV3D:
    def __init__(self, data=None, name='ProgramSV3D', point_size=1):
        data = data.view(gloo.VertexBuffer)
        program = gloo.Program(vertexPoint, fragmentSelect)
        program.bind(data)

        program['color'] = (1, 0, 0, 1)
        program['color_sel'] = 1
        program['u_model'] = np.eye(4, dtype=np.float32)
        program['u_view'] = glm.translation(0, 0, -50)
        program['a_pointSize'] = point_size

        self.name = name
        self.program = program
        self.draw_mode = gl.GL_POINTS
        self.u_model, self.u_view, self.u_projection = np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), np.eye(
            4, dtype=np.float32)
        self.lat, self.lon, self.yaw = 0, 0, 0

    def global_position(self, lat=0, lon=0, yaw=0):
        self.lat, self.lon, self.yaw = lat, lon, yaw

        glm.rotate(self.u_model, 180, 0, 1, 0)
        glm.rotate(self.u_model, yaw, 0, 0, -1)
        glm.rotate(self.u_model, lat, -1, 0, 0)
        glm.rotate(self.u_model, lon, 0, 1, 0)

    def offset_position(self, ecef):
        glm.translate(self.u_model, ecef[1], ecef[2], ecef[0])
        #print(np.dot(model, model2))
        #print(np.dot(model2, model))
        #print(self.u_model)

    def info_3d_offs(self):
        glm.rotate(self.u_model, 180, 0, 1, 0)

    def add_sv3d(self, sv3d):
        sv3d = sv3d.view(gloo.VertexBuffer)
        self.program.bind(sv3d)


class ProgramPlane:
    def __init__(self, data=None, name='ProgramPlane', face=None):
        data = data.view(gloo.VertexBuffer)
        program = gloo.Program(vertexPoint, fragmentSelect)
        program.bind(data)

        program['color'] = (1, 0, 0, 1)
        program['color_sel'] = 1
        program['u_model'] = np.eye(4, dtype=np.float32)
        program['u_view'] = glm.translation(0, 0, -50)

        self.name = name
        self.program = program
        self.draw_mode = gl.GL_TRIANGLES
        self.u_model, self.u_view, self.u_projection = np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), np.eye(
            4, dtype=np.float32)
        self.lat, self.lon, self.yaw = 0, 0, 0
        self.face = face.view(gloo.IndexBuffer)


class ProgramSV3DRegion:
    def __init__(self, data=None, name='ProgramSV3DRegion', point_size=1, anchor_matrix=np.eye(4, dtype=np.float32)):
        self.data = data.view(gloo.VertexBuffer)
        self.anchor_matrix = anchor_matrix

        program = gloo.Program(vertexPoint, fragmentSelect)
        program.bind(self.data)

        program['color'] = (1, 0, 0, 1)
        program['color_sel'] = 1
        program['a_pointSize'] = point_size
        program['u_model'] = np.eye(4, dtype=np.float32)
        program['u_view'] = np.eye(4, dtype=np.float32)

        self.name = name
        self.program = program
        self.draw_mode = gl.GL_POINTS
        self.u_model, self.u_view, self.u_projection = np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), np.eye(
            4, dtype=np.float32)

    def apply_anchor(self):
        self.data['a_position'] = base_process.sv3d_apply_m4(data=self.data['a_position'], m4=np.linalg.inv(self.anchor_matrix))



class GpyWindow:
    def __init__(self):
        self.programs = []

        window = app.Window(1024, 1024, color=(0, 0, 0, 1))
        framebuffer = np.zeros((window.height, window.width * 3), dtype=np.uint8)

        self.deg_x, self.deg_y, self.mov_x, self.mov_y, self.size, self.zoom, self.radius = 0, 0, 0, 0, 1, -200, 200
        self.u_model, self.u_view, self.u_projection = np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), np.eye(
            4, dtype=np.float32)

        self.u_view = glm.translation(0, 0, -25)

        @window.event
        def on_draw(dt):
            window.clear()
            for program_object in self.programs:
                program = program_object.program

                model = matrix_model(np.copy(program_object.u_model))
                program['u_model'] = model

                program['u_view'] = self.u_view
                program['u_projection'] = self.u_projection
                if program_object.draw_mode == gl.GL_TRIANGLES:
                    program.draw(program_object.draw_mode, program_object.face)
                else:
                    program.draw(program_object.draw_mode)
                #program.draw(gl.GL_POINTS)

        @window.event
        def on_resize(width, height):
            ratio = width / float(height)
            self.u_projection = glm.perspective(75.0, ratio, 1, 10000.0)

        @window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)

        @window.event
        def on_mouse_scroll(x, y, dx, dy):
            if self.size + dy * 0.1 < 0.1:
                self.size = 0.1
            else:
                self.size += dy * 0.1
                # self.zoom += dy*1
                # self.program['u_view'] = glm.translation(0, 0, self.zoom)

        @window.event
        def on_mouse_drag(x, y, dx, dy, button):
            if button == 2:
                self.deg_y += dy  # degrees
                self.deg_x += dx  # degrees
            elif button == 8:
                self.mov_y += dy/10  # degrees
                self.mov_x += dx/10  # degrees

        @window.event
        def on_key_press(symbol, modifiers):
            """

            :param symbol:
            :param modifiers:
            :return:
            """
            '''
            if symbol == 88:  # x
                self.view_orth_vector = np.array([self.radius, 0, 0])
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1],
                                                         self.view_orth_vector[2])
            elif symbol == 89:  # y
                self.view_orth_vector = np.array([0, self.radius, 0])
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1],
                                                         self.view_orth_vector[2])
            elif symbol == 90:  # z
                self.view_orth_vector = np.array([0, 0, self.radius])
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1],
                                                         self.view_orth_vector[2])
            elif symbol == 70:  # f
                self.view_orth_vector = -self.view_orth_vector
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1],
                                                         self.view_orth_vector[2])
            elif symbol == 80:  # p
                gl.glReadPixels(0, 0, window.width, window.height,
                                gl.GL_RGB, gl.GL_UNSIGNED_BYTE, framebuffer)
                png.from_array(framebuffer, 'RGB').save('screenshot.png')

            print('Key pressed (symbol=%s, modifiers=%s)' % (symbol, modifiers))
            # self.program['color_sel'] = 1 - self.program['color_sel']
            '''

        def matrix_model(model):
            glm.scale(model, self.size, self.size, self.size)
            glm.rotate(model, self.deg_y, 1, 0, 0)
            glm.rotate(model, self.deg_x, 0, 1, 0)
            glm.translate(model, self.mov_x, -self.mov_y, 0)
            # model[3,3] = 1
            return model

    def add_program(self, program):
        self.programs.append(program)

    @staticmethod
    def run():
        app.run()


class Program:
    deg_x, deg_y, size, zoom, radius = 0, 0, 1, -200, 200
    datas, programsdatas = [], []

    def __init__(self, _vertex, _fragment):
        self.program = gloo.Program(_vertex, _fragment)
        self.program_axis = gloo.Program(vertex, _fragment)

        position = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        color = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])

        data = np.zeros((len(position)), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
        data['a_position'] = position * 100
        data['a_color'] = color

        data = data.view(gloo.VertexBuffer)
        self.program_axis.bind(data)

    def addPoint(self, data):
        data = data.view(gloo.VertexBuffer)
        self.datas.append(data)
        self.program.bind(data)

    def widowSetting(self):
        window = app.Window(1024, 1024, color=(0, 0, 0, 1))
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
            # glm.translate(model, -self.deg_x/100, -self.deg_y/100, 0)
            # model[3,3] = 1
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
            self.size += dy * 0.1
            if self.size < 0:
                self.size = 0.1
                # self.zoom += dy*1
                # self.program['u_view'] = glm.translation(0, 0, self.zoom)

        @window.event
        def on_mouse_drag(x, y, dx, dy, button):
            self.deg_y += dy  # degrees
            self.deg_x += dx  # degrees

        @window.event
        def on_key_press(symbol, modifiers):
            if symbol == 88:  # x
                self.view_orth_vector = np.array([self.radius, 0, 0])
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1],
                                                         self.view_orth_vector[2])
            elif symbol == 89:  # y
                self.view_orth_vector = np.array([0, self.radius, 0])
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1],
                                                         self.view_orth_vector[2])
            elif symbol == 90:  # z
                self.view_orth_vector = np.array([0, 0, self.radius])
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1],
                                                         self.view_orth_vector[2])
            elif symbol == 70:  # f
                self.view_orth_vector = -self.view_orth_vector
                self.program['u_view'] = glm.translation(self.view_orth_vector[0], self.view_orth_vector[1],
                                                         self.view_orth_vector[2])
            elif symbol == 80:  # p
                gl.glReadPixels(0, 0, window.width, window.height,
                                gl.GL_RGB, gl.GL_UNSIGNED_BYTE, framebuffer)
                png.from_array(framebuffer, 'RGB').save('screenshot.png')

            print('Key pressed (symbol=%s, modifiers=%s)' % (symbol, modifiers))
            # self.program['color_sel'] = 1 - self.program['color_sel']

        self.program['color'] = (1, 0, 0, 1)
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

vertexSimple = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 a_position;      // Vertex position
attribute vec3 a_color;
varying vec3 v_color;
void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
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

vertexPoint = """
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
    gl_PointSize = a_pointSize;
    v_color = a_color;
}
"""
vertexTri = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 a_position;      // Vertex position
attribute vec3 a_color;
varying vec3 v_color;
void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    v_color = a_color;
}
"""
fragmentSelect = """
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