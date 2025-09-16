import ctypes
import numpy as np
import pandas as pd
import time
import random

# Load DLL
lib = ctypes.CDLL('./transformer.dll')

# set argument and return for cu code
lib.example.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]

lib.initWeight128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]

lib.attentionScore.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]

seq = 2

weight = np.load("weight.npy")
wQ = weight[:, 0:16]
wK = weight[:, 16:32]
wV = weight[:, 32:48]


print(wV.shape)

wQ = wQ.flatten()
wK = wK.flatten()
wV = wV.flatten()

X = []
for i in range(128 * seq):
    X[i] = random.uniform(0, 1)

lib.initWeight128(wQ, wK, wV)






