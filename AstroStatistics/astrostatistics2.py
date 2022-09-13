import numpy as np
from scipy.integrate import quad
from matplotlib import cm
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
###1

# when errors are gaussian, we can use gaussian probability function. then, can use likelihood function of gaussian.
# to make maximize likelihood function of gaussian, we have to minimize chi2
# so Least-square approach is valid approach.

###3
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

###4

## dL function shape(len(Om),len(Ol),len(zcmb)) 
# def dL(zcmb,Om,Ol):
#     result = np.zeros((len(Om),len(Ol),len(zcmb)))
#     for i,ii in enumerate(Om):
#         for j,jj in enumerate(Ol):
#             for k,kk in enumerate(zcmb):
#                 f = lambda z : 1/(np.sqrt(ii*(1+z)**3 + (1-ii-jj)*(1+z)**2 + jj))
#                 intf , err = quad(f,0,kk)
#                 result[i,j,k] = ((1+kk)*c/H0)*intf
#     return result

## dL function standard
def dL(zcmb,Om,Ol):
    result = np.zeros(len(zcmb))
    for ii in range(len(zcmb)):
        f = lambda z : 1/np.sqrt(Om*(1+z)**3 + (1-Om-Ol)*(1+z)**2 + Ol)
        intf , err = quad(f,0,zcmb[ii])
        result[ii] = ((1+zcmb[ii])*c/H0)*intf
    return result

## dL function calculated using Maclaurin's series
# def dL(zcmb,Om,Ol):
#     return (1+zcmb)*c*2*(np.sqrt((2+Om-2*Ol)*zcmb + 1)-1)/((2+Om-2*Ol)*H0)

def mu(zcmb,Om,Ol,MB):
    return 5*np.log10(dL(zcmb,Om,Ol)) + 25 + MB

def mucdm(zcmb,Om,Ol,MB):
    return LambdaCDM(70, Om, Ol).distmod(zcmb).value + MB

# %timeit mu(zcmb,0.3,0.7,0)
# %timeit mucdm(zcmb,0.3,0.7,0)

###5
# x0 = np.linspace(0,2.5)

# fig, ax = plt.subplots()

# ax.plot(zcmb,mb,".",markersize = 4, label ="data")
# ax.plot(x0,mu(x0,0.3,0.7,0), color = "b", label = "mu")
# ax.plot(x0,mucdm(x0,0.3,0.7,0), color = "r", label = "mucdm")
# ax.set_xlabel("question (5) result")
# ax.legend()
# plt.show()

###6,7

Om = np.linspace(0.1,1,100)
Ol = np.linspace(0.1,1,100)
MB = -20
chi2 = 1/2*(((mb-mu(zcmb,Om,Ol,MB))/dmb)**2).sum(axis = -1)
# use data = mb , model = 5*np.log10(dL(zcmb,Om,Ol)) + 25 + MB
# make chi2

# from matplotlib import cm
# cmap = cm.get_cmap("Blues_r",30)

# fig, ax = plt.subplots()
# img = ax.imshow(np.log(chi2), interpolation = 'bilinear', cmap = cmap, extent = [0.1,1,0.1,1], aspect = 'auto')
# fig.colorbar(img, ax = ax)
# ax.set_xlabel("question (7) result")

# plt.show()

###8

def residual(theta,zcmb):
    Om = theta[0]
    Ol = theta[1]
    MB = theta[2]
    return 1/2*(((mb-mu(zcmb,Om,Ol,MB))/dmb)**2).sum()

from scipy.optimize import minimize

res = minimize(residual,x0=(0,0,-20),args=(zcmb))
print(res)

###9

# fig, ax = plt.subplots(1)
# ax.plot(zcmb,(mb-mu(zcmb,0.3,0.68,-20)),".")
# plt.show()

###10

b = open("sys_full_long.txt","r")
line_b = b.readlines()[1:]
cov_sys = np.zeros_like(line_b, dtype=float)

for j,k in enumerate(line_b):
    cov_sys[j] = float(k)

cov_sys = cov_sys.reshape((1048,1048))

cov = np.diag(dmb**2) + cov_sys

#visualize matrix
# fig = plt.figure(figsize=(19,12))
# ax = fig.subplots(3)
# ax[0].matshow(np.diag(dmb**2))
# ax[0].set_ylabel("diag(dmb**2)")

# ax[1].matshow(cov_sys)
# ax[1].set_ylabel("cov_sys")

# ax[2].matshow(cov)
# ax[2].set_ylabel("diag(dmb**2) + cov_sys")
# ax[2].set_xlabel("question (10) result")
# plt.show()


###11

# def residual_cov(theta,zcmb):
#     Om = theta[0]
#     Ol = theta[1]
#     MB = theta[2]
#     return (((mb- mu(zcmb,Om,Ol,MB))**2)*cov).sum(axis = -1)

# res_cov = minimize(residual_cov, x0=(0,0,-20),args=(zcmb))
# print(res_cov)

del_y = mb-mu(zcmb,Om,Ol,MB)


# chi_2 = del_y@cov@del_y.T
print(del_y.shape,cov.shape,del_y.T.shape)
