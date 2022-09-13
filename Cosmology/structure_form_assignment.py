import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

GG = 1

rho0c = 1.5 * Om0 * H0**2  / aa(tt)



def  HH(aa,Om0,H0):
    Ol0 = 1-Om0
    return H0 * np.sqrt(Om0/aa**3+Ol0)


def aa(tt):
    return 


def fun(tt,yy):
    delta = yy[1]
    uu = yy[0]
    udot = -2 * HH * uu - 1.5 * Om0 * H0**2 * delta / aa(tt)**3#-2 * HH * u - 4 * np.pi * GG * rhobar * delta 
    return udot, uu


def rhobar(t,Om0):
    return Om0 *rho0c / (aa(t))**3
