import time
import fastmunk
import munkres
import numpy as np

mat = np.array([
    [12, 9, 27, 10, 23],
    [7, 13, 13, 30, 19],
    [25, 18, 26, 11, 26],
    [9, 28, 26, 23, 13],
    [16, 16, 24, 6, 9]],
    dtype=np.float64,
)


m = fastmunk.FastMunk()
start = time.time()
for _ in range(10000):
    indices = m.compute(mat)
print("FastMunk: ", time.time() - start)

m = munkres.Munkres()
start = time.time()
for _ in range(10000):
    indices = m.compute(mat)
print("Munkres: ", time.time() - start)
