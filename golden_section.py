import math
import sys

class Function(object):
    def __init__(self):
        super().__init__()
    
    def lambda_func(self, x):
        y = math.pow(x,2)-2*x
        return y


def GoldenSectionRecu(Function, LeftBound, RightBound, LastPoint, 
                    LastStride, Mode = 1, Alpha=(math.sqrt(5)-1)/2, Thresh=1e-12):
    if math.isclose(LeftBound,RightBound,abs_tol=Thresh):
        Function._x = LeftBound
        Function._y = Function.lambda_func(LeftBound)
        return
    now_stride = Alpha * LastStride 
    if Mode == 0: #left search, regard LastPoint as lamb
        lamb = RightBound - now_stride
        mu = LastPoint
    else: #right search, regard LastPoint as mu
        mu = LeftBound + now_stride
        lamb = LastPoint

    flamb = Function.lambda_func(lamb)
    fmu = Function.lambda_func(mu)
    if flamb > fmu:
        GoldenSectionRecu(Function, lamb, RightBound, mu, now_stride, Mode=1)
    else:
        GoldenSectionRecu(Function, LeftBound, mu, lamb, now_stride, Mode=0)
    return


def GoldenSection(Function, LB, RB):
    if LB > RB:
        print("invalid LB and RB")
        sys.exit()
    stride_init = (RB - LB) * ((math.sqrt(5)-1)/2)
    lamb_init = RB - stride_init
    mu_init = LB + stride_init
    flamb_init = Function.lambda_func(lamb_init)
    fmu_init = Function.lambda_func(mu_init)
    if flamb_init > fmu_init:
        GoldenSectionRecu(Function, lamb_init, RB, mu_init, stride_init, Mode=1)
    else:
        GoldenSectionRecu(Function, LB, mu_init, lamb_init, stride_init, Mode=0)
    return

if __name__ == "__main__":
    f = Function()
    GoldenSection(f,-1,1e6)
    print(f._x,f._y)