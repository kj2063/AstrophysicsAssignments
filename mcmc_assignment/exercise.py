import numpy as np
from scipy.integrate import quad
from matplotlib import cm
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM, FlatLambdaCDM, FlatwCDM
from scipy.optimize import minimize

a = open("lcparam_full_long.txt", "r")

line = a.readlines()[1:]

zcmb = np.zeros(len(line))
mb= np.zeros(len(line))
dmb = np.zeros(len(line))

for i,l in enumerate(line):
    zcmb[i] = float(l.split(" ")[1])
    mb[i] = float(l.split(" ")[4])
    dmb[i] = float(l.split(" ")[5])

h = 0.7
H0 = 100*h # km s^-1 Mpc^-1
c = 3e5 # km s^-1


def mucdm(zcmb,Om,Ol,MB):
    return LambdaCDM(H0=70,Om0=Om,Ode0=Ol).distmod(zcmb).value + MB

def mufcdm(zcmb,Om,MB):
    return FlatLambdaCDM(70, Om).distmod(zcmb).value + MB

def mufwcdm(zcmb,Om, w, MB):
    return FlatwCDM(70,Om,w).distmod(zcmb).value + MB


b = open("sys_full_long.txt","r")
line_b = b.readlines()[1:]
cov_sys = np.zeros_like(line_b, dtype=float)

for j,k in enumerate(line_b):
    cov_sys[j] = float(k)

cov_sys = cov_sys.reshape((1048,1048))

cov = np.diag(dmb**2) + cov_sys

def chi2_mucdm(par,zcmb,cov):
    Om = par[0]
    Ol = par[1]
    MB = par[2]
    print(mucdm(zcmb,Om,Ol,MB).T.shape)
    print(mucdm(zcmb,Om,Ol,MB).shape)
    return mucdm(zcmb,Om,Ol,MB).T@np.linalg.inv(cov)@mucdm(zcmb,Om,Ol,MB)

# res = minimize(chi2_mucdm,x0 = (0,0,-20),args=(zcmb,cov))

# chi2_mucdm(par,zcmb,cov)
print(mucdm(zcmb,Om,Ol,MB).T.shape)
print(mucdm(zcmb,Om,Ol,MB).shape)
