import numpy as np
from scipy.optimize import minimize
from scipy.odr import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import math

x00 = np.linspace(-12,-8,100000)
xpt = np.array([-12,-8])
k = open("RAR_SPARC.dat","r")
a = k.readlines()
k.close()
sigy =[]
sigx =[]
x =[]
y =[]
for i in range(1,len(a)):
    b= a[i].split()
    x.append(float(b[0]))
    sigx.append(float(b[1]))
    y.append(float(b[2]))
    sigy.append(float(b[3]))



a0=1.2*(10)**(-10)
def f(p,x):
    xx=np.power(10.0,x)
    z=xx/a0
    A0 = p[0]*(1+0.5*p[0])/(1+p[0])/z
    B0 = (1 +p[0])/z
    res1 = np.log10(0.5-A0+np.power(np.power(0.5-A0,2)+B0,0.5))+x
    return res1
linear = Model(f)
mydata = RealData(x, y, sx=sigx, sy=sigy)
myodr = ODR(mydata, linear, beta0=[0.5])
myoutput = myodr.run()

p_odr = myoutput.beta
s_odr = myoutput.sd_beta
s=p_odr[0]
yodr = f(p_odr,x00)
print("odr" , s)
print("odr error" , s_odr)
#def dsq(p): # d square definition
#    xx=np.power(10,x)
#    z=xx/a0
#    res= 0.5-p[1]*(1+0.5*p[1])/(1+p[1])/z+np.power(np.power(0.5-p[1]*(1+0.5*p[1])/(1+p[1])/z,2)+(1+p[1])/z,0.5)
#    res1 = np.log10(res*xx) +p[0]
#    res2 = np.sum(((y-res1)/sigy)**2)#/sigy)**2)
#    return res2
#
#sol=minimize(dsq,x0=[0.,0.01])  # minimize the sum of squared normalized residuals
def dsq(p): # d square definition
    xx=np.power(10.0,x)
    z=xx/a0
    res= 0.5-p*(1+0.5*p)/(1+p)/z+np.power(np.power(0.5-p*(1+0.5*p)/(1+p)/z,2)+(1+p)/z,0.5)
    res1 = np.log10(res*xx)
    res2 = np.sum(((y-res1)/sigy)**2)#/sigy)**2)
    return res2

sol=minimize(dsq,x0=[0.018])  # minimize the sum of squared normalized residuals
pfit=sol.x
print("pfit",pfit[0])
k=pfit[0]
xds =np.power(10.0,x00)
zds = xds/a0
yfit = np.log10((0.5 -k*(1+0.5*k)/(1+k)/zds+np.sqrt((1+k)/zds+np.power(0.5-k*(1+0.5*k)/(1+k)/zds,2)))*xds)




fig = plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(10,10)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.rc('axes',labelsize=12)

ax = plt.subplot2grid((100,200), (3, 0), rowspan=95, colspan=190)   # panel 1
ax.scatter(x,y,s=1,c='r',marker='o',alpha=1.,label='data')
ax.errorbar(x,y,xerr=sigx,yerr=sigy,c='b',fmt='none',elinewidth=1,capsize=1)
ax.plot(x00,yfit,c='r',lw='4',label='lsq fit(e = %1.4f)'%pfit[0])
ax.plot(xpt,xpt,c='black',ls='--',lw='2',label='y = x')
ax.plot(x00,yodr,c='orange',ls='--',lw='4',label='odr fit(e = %1.4f, e_err = %1.4f)'%(p_odr[0],s_odr[0]))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('linearooo relation',fontsize='25')
#ax.set_xlim(-12,-8)
#ax.set_ylim(-12,-8)
ax.legend(fontsize=10)

plt.show()
plt.close(fig)
