import numpy as np
from scipy import spatial

x, y = np.mgrid[0:4, 0:4]
points = zip(x.ravel(), y.ravel())
tree = spatial.KDTree(points)
tree.query_ball_point([2, 0], 1)