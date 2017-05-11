#!/usr/bin/python3
# ==============================================================
# blending
# ==============================================================
import numpy as np
import scipy.misc


def myNorm(wight_list):
    wight_sum = sum(wight_list)
    if wight_sum == 0:
        return np.zeros(len(wight_list))
    else:
        wight_list_norm = [wight / wight_sum for wight in wight_list]
        return np.array(wight_list_norm)

test_num = 5
dist_list = [19.4169833713, 8.33688895814, 7.07106781187, 16.9071588345, 23.7495991383]
wight = np.array([1. / dist for dist in dist_list])
wightnorm = myNorm(wight)

# read images
img_list = []
for i in range(0, test_num):
    img = scipy.misc.imread('{:d}.png'.format(i + 6)).astype(np.float32)
    img_list.append(img)
# blending
im_blending = np.zeros((256, 512, 3))
for y in range(0, 256):
    for x in range(0, 512):
        # remove black
        notBlack = np.ones(test_num)
        for i in range(0, test_num):
            img_pixel = img_list[i][y, x, :]
            if img_pixel[0] == 0 and img_pixel[1] == 0 and img_pixel[2] == 0:
                notBlack[i] = 0

        pixel_wight = wight * notBlack
        pixel_wight_norm = myNorm(pixel_wight)
        for i in range(0, test_num):
            img_pixel = img_list[i][y, x, :]
            im_blending[y, x, :] += img_pixel * pixel_wight_norm[i]

scipy.misc.imsave('yolo.png', im_blending)
