# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------

import scipy.signal
import numpy as np

x = np.array([[1, 2, 3],
              [7, 8, 9],
              [4, 5, 6]])
y = scipy.signal.resample(x, 2)
print(y)
