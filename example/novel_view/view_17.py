import numpy as np
import scipy.misc

img6 = scipy.misc.imread('6.png').astype(np.float32)
img7 = scipy.misc.imread('7.png').astype(np.float32)
img8 = scipy.misc.imread('8.png').astype(np.float32)

scipy.misc.imsave('blend.png', img6 * 0.33 + img7 * 0.33 + img8 * 0.33)