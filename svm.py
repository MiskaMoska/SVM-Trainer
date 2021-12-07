import numpy as np
from zoutendijk import Zoutend
from svm_params import *
from matplotlib import markers, pyplot as plt

class ObjFunc(object):

    def __init__(self):
        super().__init__()
        self.cnt = 0
    
    def func(self, x):
        temp = np.dot(MAT_A,x)
        temp = np.dot(x,temp)/2
        temp += np.dot(vec_b,x)
        return temp

    def grad(self, x):
        return np.dot(MAT_A,x) + vec_b

    def upd_state(self, x_k, d_k):
        self.x_k = x_k
        self.d_k = d_k

    def lambda_func(self, lamb):
        return self.func(self.x_k + lamb * self.d_k)



if __name__ == "__main__":
    for i in range(MAT_X.shape[0]):
        s = plt.scatter(MAT_X[i,0], MAT_X[i,1],
                            marker='.' if vec_y[i] == 1 else '.', 
                            s=50, color = 'b' if vec_y[i] == 1 else 'r')
    plt.show()

    x = np.zeros(vec_b.shape[0])
    zt = Zoutend(MAT_C,vec_c,MAT_E,vec_e,ObjFunc,x)
    zt.search()
    res = zt.opt
    alpha_y = res * vec_y
    omega = np.dot(np.transpose(MAT_X),alpha_y)
    # print(zt.x)
    sv_idx = np.where(zt.x > 1e-12) # extract support vectors
    gv_idx = np.where(np.isclose(zt.x,C,atol=1e-3)) # extract gap vectors
    sv_alpha = zt.x[sv_idx[0]]
    print("sv_alphas:",sv_alpha)
    b = vec_y[sv_idx[0]] - np.dot(MAT_X[sv_idx[0]],omega)
    # print(b)
    b_av = np.sum(b)/b.shape[0]
    # print(b_av)
    # print(omega)

    p,n = True, True
    for i in range(MAT_X.shape[0]): # plot all samples
        if vec_y[i] == 1:
            s = plt.scatter(MAT_X[i,0], MAT_X[i,1],marker='.', s=50,  # positive samples
                                color = 'b' ,label = "positive" if p else None)
            p = False
        else:
            s = plt.scatter(MAT_X[i,0], MAT_X[i,1],marker='.', s=50, # negative samples
                                color = 'r' ,label = "negative" if n else None)
            n = False

    SV_X = MAT_X[gv_idx[0],:]
    gv_vec_y = vec_y[gv_idx[0]]
    p,n = True, True
    for i in range(SV_X.shape[0]): # plot samples in soft gap
        if gv_vec_y[i] == 1:
            s = plt.scatter(SV_X[i,0], SV_X[i,1], marker='.', s=60, # positive samples
                                color = 'white',edgecolor = 'b', linewidths = 1,
                                label = "positive in gap" if p else None)
            p = False
        else:
            s = plt.scatter(SV_X[i,0], SV_X[i,1], marker='.', s=60, # negative samples
                                color = 'white',edgecolor = 'r', linewidths = 1,
                                label = "negative in gap" if n else None)
            n = False

    x1_cut = [min(L1,L2),max(R1,R2)]
    x2_cut = [-(b_av+omega[0]*min(L1,L2))/omega[1],-(b_av+omega[0]*max(R1,R2))/omega[1]]
    print(x1_cut)
    print(x2_cut)
    plt.legend()
    plt.plot(x1_cut,x2_cut,color = 'black',linewidth = 1)
    plt.show()
