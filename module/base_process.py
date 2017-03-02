#!/usr/bin/python3
import numpy as np
import math

# Great Circle
earthR = 6371000

# WGS84
a = 6378137.0    # Semi-major axis
b = 6356752.314245    # Semi-minor axis
ee = 0.00669437999014    # First eccentricity squard
ee_ = 0.00673949674228    # Second eccentricity squared


def gps_2_gl_great_circle(lat, lon):
    global earthR
    lat = float(lat)
    lon = float(lon)
    wr = earthR*np.cos(lat * np.pi / 180)
    wy = earthR*np.sin(lat * np.pi / 180)
    wx = wr * np.sin(lon * np.pi / 180)
    wz = wr * np.cos(lon * np.pi / 180)
    return wx, wy, wz


def gl_2_gps_great_circle(vec):
    #[x y z] = vec
    x = vec[0]
    y = vec[1]
    z = vec[2]
    r = np.linalg.norm((x, 0, z))
    lat = math.atan(y / r) / np.pi * 180
    lon = math.atan(x / z) / np.pi * 180
    if y > 0:
        lon += 180
    return lat, lon


def pos_2_deg(x, y, z):
    lng = np.arctan(x / y) / np.pi * 180
    if x >= 0 and y >= 0:
        lng += + 180
    elif x <= 0 and y >= 0:
        lng -= 180
    r = np.linalg.norm([x, y, 0])
    lat = np.arctan(z / r) / np.pi * 180
    return lng, lat


def gl_2_ecef_great_circle(vec):
    x, y, z = vec
    r = np.linalg.norm((x, y, 0))
    lat = math.atan(z / r) / np.pi * 180
    lon = math.atan(x / y) / np.pi * 180
    if y < 0:
        if x < 0:
            lon -= 180
        else:
            lon += 180
    return lat, lon


# http://w3.uch.edu.tw/ccchang50/crd_trsnafer.pdf
# https://en.wikipedia.org/wiki/ECEF
def geo_2_ecef(lat, lon, height=0):
    lat *= np.pi / 180
    lon *= np.pi / 180
    N = a / np.sqrt(1 - ee * np.power(np.sin(lat), 2))
    X = (N + height) * np.cos(lat) * np.cos(lon)
    Y = (N + height) * np.cos(lat) * np.sin(lon)
    Z = (N * (1-ee) + height) * np.sin(lat)
    return np.array([X, Y, Z])


def ecef_2_geo(X, Y, Z):
    p = np.sqrt(X*X + Y*Y)
    theta = np.arctan((Z * a) / (p * b))
    lat = np.arctan((Z + ee_ * b * np.power(np.sin(theta), 3)) 
                   / (p - ee * a * np.power(np.cos(theta), 3)))
    lon = np.arctan2(Y, X)
    N = a / np.sqrt(1 - ee * np.power(np.sin(lat), 2))
    h = p / np.cos(lat) - N
    #print p/np.cos(lat), N
    lat = lat / np.pi * 180
    lon = lon / np.pi * 180 
    #if Y > 0:
    #    lon += 180    
    return lat, lon, h


def sv3d_apply_m4(data, m4):
    vec4 = np.hstack([data, np.ones((len(data), 1))])
    vec4_mul = np.dot(vec4, m4)
    return vec4_mul[:, 0:3]

"""
Amazing work from
http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
"""
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def create_plane(n_point=100, sx=10, sy=10, ground_z=-2):
    n_point = n_point
    sx, sy = sx, sy
    ground = np.ones((n_point, n_point)) * ground_z
    x = np.linspace(-10, 10, n_point)
    y = np.linspace(-10, 10, n_point)
    xv, yv = np.meshgrid(x, y)

    data = np.zeros((n_point, n_point), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])

    data['a_position'] = np.dstack((xv, yv, ground))
    data['a_color'] = (1, 0, 0)
    return data


def vec_2_panorama(vec):
    lng = math.atan(-vec[0] / vec[1]) / math.pi * 180
    if vec[0] >= 0 and vec[1] >= 0:
        lng += 180
    elif vec[0] <= 0 and vec[1] >= 0:
        lng -= 180
    r = np.linalg.norm(np.array((vec[0], 0, vec[1]), dtype=np.float32))
    lat = math.atan(vec[2] / r) / math.pi * 180

    return lat, lng


def bounding_box(points):
    """
    [xmin ymin]
    [xmax ymax]
    """
    a = np.zeros((2,2))
    a[0, :] = np.min(points, axis=0)
    a[1, :] = np.max(points, axis=0)
    return a