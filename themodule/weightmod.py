import numpy as np
import ctypes

w = np.random.rand(128, 48).astype(np.float32)

np.save("weight.npy", w)

qs = w.flatten()

# c_array = (ctypes.c_float * arr_flat.size)(*arr_flat)
