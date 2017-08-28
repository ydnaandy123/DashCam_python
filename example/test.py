from scipy import interpolate
import numpy as np
import scipy.ndimage
x = np.arange(0, 10)
y = scipy.ndimage.zoom(x, zoom=(2.0), mode='nearest')
print(y)
