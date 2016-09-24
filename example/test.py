import numpy as np

print("The depthMap's size of id:%s is unusual:(%d, %d)"
      % ('jhdf', 123, 456))

'''
B = np.array([[ -6.25916012e-03,   3.20910811e-01,   0.00000000e+00,
          4.66891289e+01],
       [ -2.26589158e-01,   5.09252548e-02,   0.00000000e+00,
          2.87899723e+01],
       [  0.00000000e+00,   0.00000000e+00,   1.21887244e-01,
          1.63451672e-01], 
          [0, 0, 0, 1]])

print (B)

A = np.array([[-0.47995195,  0.41581649,  0.77249128,  0.        ],
       [ 0.87729245,  0.22544707,  0.42371172,  0.        ],
       [ 0.00203044,  0.88106203, -0.47299644,  0.        ],
       [0 , 0, 0, 1]])
#A = np.transpose(A)

print (A)

testV = np.array([-72.74919557543679, 14.43233202405186, 1.194673749553661e-15, 1])

print (testV)
testV = np.dot(B[0:3, 0:4], testV)
print (testV)
testV = np.hstack((testV, 1))
print (testV)
testV = np.dot(A[0:3, 0:4], testV)
print (testV)
'''