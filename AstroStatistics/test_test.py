import numpy as np
from scipy.integrate import quad
from sympy import symbols, integrate
import matplotlib.pyplot as plt

Om = np.linspace(0.1,1,100)[:,np.newaxis,np.newaxis]
Ol = np.linspace(0.1,1,100)[np.newaxis,:,np.newaxis]

a = open("lcparam_full_long.txt", "r")

line = a.readlines()[1:]

zcmb = np.zeros(len(line))
mb= np.zeros(len(line))
dmb = np.zeros(len(line))

for q,l in enumerate(line):
    zcmb[q] = float(l.split(" ")[1])
    mb[q] = float(l.split(" ")[4])
    dmb[q] = float(l.split(" ")[5])

h = 0.7
H0 = 100*h # km s^-1 Mpc^-1
c = 3e5 # km s^-1

def dL(zcmb,Om,Ol):
    result = np.zeros((len(Om),len(Ol),len(zcmb)))
    for i,ii in enumerate(Om):
        for j,jj in enumerate(Ol):
            for k,kk in enumerate(zcmb):
                f = lambda z : 1/(np.sqrt(ii*(1+z)**3 + (1-ii-jj)*(1+z)**2 + jj))
                intf , err = quad(f,0,kk)
                result[i,j,k] = ((1+kk)*c/H0)*intf
    return result

def dl(zcmb,Om,Ol):
    return (1+zcmb)*c*2*(np.sqrt((2+Om-2*Ol)*zcmb + 1)-1)/((2+Om-2*Ol)*H0)

def mu(zcmb,Om,Ol,MB):
    return 5*np.log10(dl(zcmb,Om,Ol)) + 25 + MB

# mb_arr = np.array(list(mb)*(len(Om)*len(Ol))).reshape((len(Om),len(Ol),len(mb)))
# dmb_arr = np.array(list(dmb)*(len(Om)*len(Ol))).reshape((len(Om),len(Ol),len(dmb)))

MB = 0
chi2 = 1/2*(((mb-mu(zcmb,Om,Ol,MB))/dmb)**2).sum(axis = -1)

# def chi2(Om,Ol,MB):
#     return 1/2*(((mb_arr-mu(zcmb,Om,Ol)-MB)/dmb_arr)**2).sum(axis=-1)

# def residual(theta,zcmb):
#     Om = theta[0]
#     Ol = theta[1]       
#     MB = theta[2]
#     return 1/2*(((mb-mu(zcmb,Om,Ol,MB))/dmb)**2).sum(axis = -1)

# from scipy.optimize import minimize

# res = minimize(residual, x0 = (1,1,40),args=(zcmb))
# print(res.x)

from matplotlib import cm
cmap = cm.get_cmap("Blues_r",30)

fig, ax = plt.subplots()
img = ax.imshow(np.log(chi2), interpolation = 'bilinear', cmap = cmap, extent = [0.1,1,0.1,1], aspect = 'auto')
fig.colorbar(img, ax = ax)

plt.show()

# def residual(theta,zcmb):
#     Om = theta[0]
#     Ol = theta[1]
#     MB = theta[2]
#     return 1/2*(((mb_arr-mu(zcmb,Om,Ol)-MB)/dmb_arr)**2).sum(axis=-1)

# from scipy.optimize import minimize
# res = minimize(residual,x0=(1,1,1),args=(zcmb))
# print(res.x)

# def chi2(zcmb,Om,Ol):
#     mb_arr = mb[np.newaxis,np.newaxis,:]
#     dmb_arr = dmb[np.newaxis,np.newaxis,:]


# a = open("lcparam_full_long.txt", "r")

# line = a.readlines()[1:]

# zcmb = np.zeros(len(line))
# mb= np.zeros(len(line))
# dmb = np.zeros(len(line))

# for i,l in enumerate(line):
#     zcmb[i] = float(l.split(" ")[1])
#     mb[i] = float(l.split(" ")[4])
#     dmb[i] = float(l.split(" ")[5])

# h = 0.7
# H0 = 100*h # km s^-1 Mpc^-1
# c = 3e5 # km s^-1

# def dL(zcmb,Om,Ol,MB):
#     result = np.zeros(len(line))
#     for ii in range(len(line)):
#         f = lambda z : 1/(np.sqrt(Om*(1+z)**3 + (1-Om-Ol)*(1+z)**2 + Ol))
#         intf , err = quad(f,0,zcmb[ii])
#         result[ii] = ((1+zcmb[ii])*c/H0)*intf
#     return result

# print(dL(zcmb,Om,Ol,MB))