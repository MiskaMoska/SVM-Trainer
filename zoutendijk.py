import numpy as np
import math
import sys
from numpy.core.numeric import isclose
from scipy import optimize
from golden_section import GoldenSection

class ObjFunc(object):
    def __init__(self):
        super().__init__()
        self.cnt = 0
    
    def func(self, x):
        y = math.pow(x[0],2) + 2*math.pow(x[1],2) + 3*math.pow(x[2],2)\
            + x[0]*x[1] - 2*x[0]*x[2] + x[2]*x[1] - 4*x[0] - 6*x[1]
        return y

    def grad(self, x):
        return np.array([2*x[0] + x[1] - 2*x[2] -4,
                        x[0] + 4*x[1] + x[2] -6,
                        -2*x[0] + x[1] + 6*x[2]])

    def upd_state(self, x_k, d_k):
        self.x_k = x_k
        self.d_k = d_k

    def lambda_func(self, lamb):
        return self.func(self.x_k + lamb * self.d_k)




class Zoutend(object):
    '''
    Zoutendijk Method
    min f(x)
    s.t. Ax >= b, Ex = e
    where:
    A is a matrix
    b is a vector
    E is a matrix
    e is a vector
    f is object function, and is a class object
    '''

    gen_cnt = 0

    def __init__(self,A,b,E,e,f,init_x):
        '''
        Initialize Zoutend Parameters
        init_x is the initial x, and is a vector
        '''
        super().__init__()
        self.A = A
        self.b = b
        self.E = E
        self.e = e
        self.f = f() #instantiate a ObjFunc instance
        self.x = init_x
        self.y = None #opt f(x) value in every iteration
        self.opt = None
        self.init_x = init_x
        Zoutend.gen_cnt += 1
        # print("a zoutend object generated")
        # print("starting point:",self.x)

    def opt_d(self):
        '''
        This method optimizes and returns search direction "d"
        '''
        idx_1, idx_2 = [],[]
        for i in range(self.A.shape[0]):
            if math.isclose(np.dot(self.A[i],self.x),self.b[i],abs_tol=1e-3):
            # if np.dot(self.A[i],self.x) == self.b[i]:
                idx_1.append(i)
            else:
                idx_2.append(i)
        self.A1 = self.A[idx_1,:] #active constraint at self.x
        self.A2 = self.A[idx_2,:] #loosen constraint at self.x
        self.b1 = self.b[idx_1] #active constraint at self.x
        self.b2 = self.b[idx_2] #loosen constraint at self.x
        self.y = self.f.func(self.x)
        # print("x:",self.x)
        # print("f(x):",self.f.func(self.x))
        # print("grad:",self.f.grad(self.x))
        # print("A1:",self.A1)
        # print("b1:",self.b1)

        eye = np.eye(self.x.shape[0])
        conc_eye = np.concatenate([eye,-eye],axis=0)
        uni_vec = np.ones(2*self.x.shape[0])
        Aub = np.concatenate([-self.A1,conc_eye],axis=0)
        bub = np.concatenate([np.zeros_like(self.b1),uni_vec],axis=0)
        # print("Aub:",Aub)
        # print("bub:",bub)

        res = optimize.linprog(self.f.grad(self.x), A_ub=Aub, b_ub=bub, A_eq=self.E, 
                                b_eq=None if self.e == None else np.zeros_like(self.e),
                                method = 'simplex', bounds=(None,None), options=None, callback=None)
        
        if res["success"] == False:
            print("LP process error at x =",self.x)
            sys.exit()
        # print("zoutend condition:",res["fun"])
        # print(np.linalg.norm(self.f.grad(self.x)))
        # print("init_x:",self.init_x)
        if math.isclose(res["fun"],0,abs_tol=1e-05):
            self.opt = self.f.func(self.x)
            return True
        print("Zoutend LP Opt Value:",res["fun"])
        self.d = res["x"]
        return False

    def opt_lambda(self):
        self.f.upd_state(self.x,self.d)
        self.d_bar = np.dot(self.A2,self.d)
        idx = np.where(self.d_bar < 0)
        if idx[0].shape[0] == 0: #no d_bar_j < 0
            GoldenSection(self.f,0,1e6)
        else:
            self.b_bar = self.b2 - np.dot(self.A2, self.x)
            lambda_max = np.min(self.b_bar[idx[0]]/self.d_bar[idx[0]])
            GoldenSection(self.f,0,lambda_max)
        # self.fvalue = self.f._y
        self.x += self.f._x * self.d

    def search(self):
        n = 1
        while True:
            if self.opt_d():
                break
            self.opt_lambda()
            n += 1
            # print(n)
        return


if __name__ == "__main__":
    A = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,-2,-1]])
    b = np.array([0,0,0,-4])
    init_x = np.array([0,0,0],dtype=float)
    E = None
    e = None
    zt = Zoutend(A,b,E,e,ObjFunc,init_x)
    zt.search()




