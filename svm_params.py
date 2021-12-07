from types import resolve_bases
import pandas as pd
import numpy as np
import math
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt

C = 0.03

L1=0
R1=15
D1=0
U1=10

L2=11
R2=28
D2=20
U2=40

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
eigenvalue, _ = np.linalg.eig(MAT_A)
# print(eigenvalue)
# print(MAT_A)

vec_b = -np.ones_like(vec_y)

MAT_E = vec_y.reshape(1,vec_y.shape[0])
vec_e = np.array([0])

MAT_C = np.concatenate([np.eye(vec_y.shape[0]),-np.eye(vec_y.shape[0])],axis=0)
vec_c = -C * np.concatenate([np.zeros_like(vec_y),np.ones_like(vec_y)],axis=0)
