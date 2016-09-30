import numpy as np

A = np.array([[0, 1, np.nan], [3, 4, 5], [6, 7, 8]], dtype=np.float32)

A = A.flatten()
print(A)

#print(A[~(A > 3) * (A > 0)])