#!/usr/bin/python3
import sys
sys.path.append('module')	# use the module under Module
import glumpy_setting
import google_parse
import numpy as np
fileID = 'test'
streetView3DRegion = google_parse.StreetView3DRegion(fileID)
topology = streetView3DRegion.createTopoloy()
print (topology)
from time import clock
from scipy import spatial

def test():
    K = 11
    ndata = 10000
    ndim = 12
    data =  10 * np.random.rand(ndata*ndim).reshape((ndim,ndata) )
    #knn_search(data, K)

if __name__ == '__main__':
    t0 = clock()
    test()
    t1 = clock()
    print ("Elapsed time:", t1-t0)

#data = np.zeros((100), dtype = [('a_position', np.float32, 3), ('a_color', np.float32, 3)])

#streetView3D = glumpy_setting.ProgramSV3D(topology)
#streetView3D.widowSetting()
