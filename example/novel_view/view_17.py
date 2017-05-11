import numpy as np
import scipy.misc

img6 = scipy.misc.imread('6.png').astype(np.float32)
img7 = scipy.misc.imread('7.png').astype(np.float32)
img8 = scipy.misc.imread('8.png').astype(np.float32)

# dist wight
#im_blending = img6 * 0.33 + img7 * 0.33 + img8 * 0.33

# remove black
im_blending = np.zeros((256, 512, 3))
for i in range(0, 256):
    for j in range(0, 512):
        if img6[i, j, 0] == 0 and img6[i, j, 1] == 0 and img6[i, j, 2] == 0:
            im_blending[i, j, :] = img7[i, j, :] * 0.5 + img8[i, j, :] * 0.5
        elif img7[i, j, 0] == 0 and img7[i, j, 1] == 0 and img7[i, j, 2] == 0:
            im_blending[i, j, :] = img6[i, j, :] * 0.5 + img8[i, j, :] * 0.5
        elif img8[i, j, 0] == 0 and img8[i, j, 1] == 0 and img8[i, j, 2] == 0:
            im_blending[i, j, :] = img6[i, j, :] * 0.5 + img7[i, j, :] * 0.5
        else:
            im_blending[i, j, :] = img6[i, j, :] * 0.33 + img7[i, j, :] * 0.33 + img8[i, j, :] * 0.33



#scipy.misc.imsave('blend.png', img6 * 0.33 + img7 * 0.33 + img8 * 0.33)
scipy.misc.imshow(im_blending)