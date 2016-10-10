import numpy as np
from scipy import spatial

points = [[0, 0, 0],
          [1, 0, 0],
          [10, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]
tree = spatial.KDTree(points)

points = [[0, 0, 0], [1, 1, 1]]
other = spatial.KDTree(points)

yo = (tree.count_neighbors(other=other, r=1.5))
print(yo)