import numpy as np
b =  (np.random.random_integers(5,10,(5,3)))
a = np.zeros((3,3,3))

a[0:2,2,:] = b[3,:]

print (a) 
