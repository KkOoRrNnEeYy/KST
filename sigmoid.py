import numpy as np
from numpy import sin, cos, pi, exp, log, e


class PSI_sigmoid:
    
    def __init__(self, k = 2 * (e + 1) / (e - 1)):
        self.k = k
    
    def psi(self, x):
        integer = 0
        if np.abs(x) > 1:
            integer = int(x)
            x -= int(x)
        dec = self.k * (1 / (1 + exp(-x)) - 0.5)
        return integer + dec
    
    def d1psi(self, x):
        if np.abs(x) > 1:
            x -= int(x)
        return exp(-x) * self.k / (1+exp(-x))**2

    def d2psi(self, x):
        if np.abs(x) > 1:
            x -= int(x)
        return exp(x)*(exp(x)-1)*self.k / (exp(x) + 1)**3

    def d3psi(self, x):
        if np.abs(x) > 1:
            x -= int(x)
        return exp(x)*(1-4*exp(x)+exp(2*x))*self.k / (1+exp(x))**4

    def psi_(self, x):
        integer = 0
        if np.abs(x) > 1:
            integer = int(x)
        x -= int(x)
        dec = -log(self.k / (self.k/2 + x) - 1)
        return integer + dec