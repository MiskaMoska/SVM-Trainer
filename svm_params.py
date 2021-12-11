from types import resolve_bases
import pandas as pd
import numpy as np
import math
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt


'''
C is quite a magical parameter, 
you can't easily figure out how it works but
here are some tips about how to choose the value of it:
------------------------------------------------------------
@1. when the quantity of samples increases, lift it up.
@2. when the two classes are difficult to divide, reduce it.
@3. when the value of samples are big, lift it up. But usually we use
    the normalized sample values so that we don't need to care about this problem.
------------------------------------------------------------
'''
C = 0.03

'''
The following patameters determine the distribution of two classes
Each four bounds determine the limiting area of the random samples of each class
'''
# Class A
L1=0 # Left bound
R1=5 # Right bound
D1=0 # Down bound
U1=5 # Up bound

# Class B
L2=3 # Left bound
R2=8 # Right bound
D2=3 # Down bound
U2=8 # Up bound

x1 = np.random.rand(100,2)
x1[:,0] = x1[:,0] * (R1-L1) + L1
x1[:,1] = x1[:,1] * (U1-D1) + D1 

x2 = np.random.rand(100,2)
x2[:,0] = x2[:,0] * (R2-L2) + L2
x2[:,1] = x2[:,1] * (U2-D2) + D2 

vec_y = np.ndarray(x1.shape[0]+x2.shape[0])
vec_y[0:x1.shape[0]] = 1
vec_y[x1.shape[0]:] = -1

MAT_X = np.concatenate([x1,x2],axis=0)

A1 = np.dot(MAT_X,np.transpose(MAT_X))
A2 = np.outer(vec_y,vec_y)
MAT_A = A1 * A2

vec_b = -np.ones_like(vec_y)

MAT_E = vec_y.reshape(1,vec_y.shape[0])
vec_e = np.array([0])

MAT_C = np.concatenate([np.eye(vec_y.shape[0]),-np.eye(vec_y.shape[0])],axis=0)
vec_c = -C * np.concatenate([np.zeros_like(vec_y),np.ones_like(vec_y)],axis=0)
