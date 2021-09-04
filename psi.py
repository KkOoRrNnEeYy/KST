import numpy as np
from math import factorial
from bisect import bisect_left

def nCk(n ,k):
    return factorial(n) / (factorial(k) * factorial(n-k))

def alpha(p, n, k, g):
    alpha = 0
    if p == 1:
        return 1
    r = 1
    while True:
        inc = g**(-(p-1) * (n**r-1)/(n-1))
        if inc < g**(-k):
            return alpha + inc
        alpha += inc
        r += 1

def beta(r):
    n = 2
    return (n**r - 1) / (n - 1)

def get_Dk(k=4, g=10):
    return np.linspace(0, 1, g**k+1)

def get_Dk_span(dk_min, dk_max, k=4, g=10):
    Dk_span = []
    h = g**(-k)
    while dk_min <= dk_max:
        Dk_span.append(dk_min)
        dk_min = round(dk_min + h, k)
    return Dk_span

def get_I(dk, k=4, g = 10):
    dk = round(dk * g**k)
    I = []
    for i in range(k):
        I.append(dk%g)
        dk = dk // g
    return I[::-1]

def get_psi_I(I, g=10):
    k = len(I)
   
    if k == 1:
        return I[0] / g
    else:
        if I[-1] != g-1:
            return get_psi_I(I[:-1], g) + I[-1] / g**(beta(k))
        else:
            I_k_ = I[:]
            I_k = I[:-1]
                        
            while I_k_[-1] == g-1:
                I_k_ = I_k_[:-1]
                if len(I_k_) == 0:
                    break
            if len(I_k_) == 0:
                I_k_ = [g]
            else:
                I_k_[-1] += 1

            return 0.5*(get_psi_I(I_k_, g) + get_psi_I(I_k, g) + (g-2)/g**(beta(k)))
            
def get_psi(dk, k=4, g=10):
    if dk < 0:
        sign = -1
        dk = -dk
    else:
        sign = 1
    integer = int(dk)
    dk -= integer
    return sign*(get_psi_I(get_I(dk, k, g)) + integer)

def get_psi_span(Dk_span, k=4, g=10):
    return [get_psi(dk, k, g) for dk in Dk_span]

def get_dpsi(dk, der_degree=1, k=4, g=10):
    h = g**(-k)
    s = 0
    for k in range(der_degree+1):
        s += (-1)**(k + der_degree) * nCk(der_degree, k) * get_psi(dk + k * h, k, g)
    return s / h**(der_degree)

def get_dpsi_span(dk_min, dk_max, der_degree=1, k=4, g=10):
    Dk = get_Dk_span(dk_min, dk_max, k, g)
    return [get_dpsi(dk, der_degree, k, g) for dk in Dk]

def calc_total_arclength(k=4, g=10):
    h = g**(-k)
    Dk = np.arange(0, 1+h, h)
    psis = [get_psi(dk, k, g) for dk in Dk]
    difs = [0, *[psis[i+1] - psis[i] for i in range(len(psis)-1)]]
    steps_len = [np.sqrt(difs[i]**2 + h**2) for i in range(len(difs))]
    return np.sum(steps_len)

def get_psi_lip(dk, total_arclength, k=4, g=10):
    if dk < 0:
        sign = -1
        dk = -dk
    else:
        sign = 1
    integer = int(dk)
    dk -= integer
    
    h = g**(-k)
    Dk = np.arange(0,dk+h/10,h)
    psis = [get_psi(dk, k, g) for dk in Dk]
    difs = [0, *[psis[i+1] - psis[i] for i in range(len(psis)-1)]]
    steps_len = [np.sqrt(difs[i]**2 + h**2) for i in range(len(difs))]
    
    cum_len = np.sum(steps_len)
    psi_lip = cum_len / total_arclength
    
    return sign * (integer + psi_sigma)


def get_psi_lip_span(Dk, k=4, g=10):
    over_range = len(Dk) - g**k - 1
    over_range_arrs = [g**k for i in range(over_range // (g**k) + 1)]
    over_range_arrs[-1] = over_range % (g**k)

    
    Dk = Dk[:g**(k)+1]
    psis = [get_psi(dk, k, g) for dk in Dk]
    h = g**(-k)
    difs = [0, *[psis[i+1] - psis[i] for i in range(len(psis)-1)]]
    steps_len = [np.sqrt(difs[i]**2 + h**2) for i in range(len(difs))]
    
    cum_len = np.cumsum(steps_len)
    psis_lip = list(cum_len / cum_len[-1])
    
    for i in range(len(over_range_arrs)):
        for j in range(over_range_arrs[i]):
            psis_lip.append(psis_lip[j]+i+1)
    
    return np.array(psis_lip)

def get_f_span(Xs, f, args=[]):
    return [f(x, *args) for x in Xs]
    
def get_derf_span(Xs, Ys):
    der_span = []
    for i in range(len(Ys)-1):
        der_span.append((Ys[i+1] - Ys[i]) / (Xs[i+1] - Xs[i]))
    return der_span

def make_function(Xs, Ys):
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    def function(x):
        ind = np.argmin(np.abs(Xs - x))
        if ind >= len(Ys):
            return Ys[-1]
        return Ys[ind]
    return function

def make_dfunction(f, step, order=1, save_mode=False):
    def dfunction(x, *args):
        s = 0
        for k in range(order+1):
            s += (-1)**(k + order) * nCk(order, k) * f(x + k * step, *args)
        if save_mode:
            if s == 0: return dfunction(x-step, *args)
        return s / step**order
    return dfunction




