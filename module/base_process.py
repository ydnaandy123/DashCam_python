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


# http://w3.uch.edu.tw/ccchang50/crd_trsnafer.pdf
# https://en.wikipedia.org/wiki/ECEF
def geo_2_ecef(lat, lon, height=0):
    lat *= np.pi / 180
    lon *= np.pi / 180
    N = a / np.sqrt(1 - ee * np.power(np.sin(lat), 2))
    X = (N + height) * np.cos(lat) * np.cos(lon)
    Y = (N + height) * np.cos(lat) * np.sin(lon)
    Z = (N * (1-ee) + height) * np.sin(lat)
    return X, Y, Z


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
