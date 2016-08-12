#!/usr/bin/python3
import numpy as np
import math

earthR = 6371000

def GPS2GL(lat, lon):
    global earthR
    lat = float(lat)
    lon = float(lon)
    wR = earthR*np.cos(lat * np.pi / 180)
    wY = earthR*np.sin(lat * np.pi / 180)
    wX = wR * np.sin(lon * np.pi / 180)
    wZ = wR * np.cos(lon * np.pi / 180)
    return (wX, wY, wZ)

def GL2GPS(vec):
    #[x y z] = vec
    x = vec[0]
    y = vec[1]
    z = vec[2]
    lon = math.atan(x / z) / np.pi * 180
    if y > 0:
        lon += 180
    r = np.linalg.norm((x, 0, z))
    lat = math.atan(y / r) / np.pi * 180
    return (lat, lon)