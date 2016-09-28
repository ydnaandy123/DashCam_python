import numpy as np

A = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float32)



TF_1 = A > 5
TF_2 = A < 7

print(TF_1, TF_2)

print(TF_1 * TF_2)

print(A[(A > 3) * (A < 5)])