vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 a_position;      // Vertex position
attribute vec3 a_color;
varying vec3 v_color;
void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    gl_PointSize = 1.0;
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
        gl_FragColor = vec4(v_color/255, 1.0);
    }
    else{
        gl_FragColor = color;
    }
}
"""