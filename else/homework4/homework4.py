import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.integrate import nquad

#####1
a = open('et.dat', 'r')
a_line = a.readlines()[1:]

e = np.zeros(len(a_line))
x = np.zeros(len(a_line))

for ii,i in enumerate(a_line):
    e[ii] = float(i.split("   ")[0])
    x[ii] = float(i.split("   ")[1])

e_median = np.median(e)
e_mean = np.mean(e)

sigma_mu = np.sqrt(np.sum((e-e_mean)**2)/(len(e)*(len(e)-1)))

print("1.(1) e_median = ",e_median)
print("      standard deviation of the mean = ",sigma_mu,"\n")


n = 1000

med_bost = np.zeros(n)

for i in range(n):
    xi = np.random.randint(len(e),size = len(e))
    sample_bost = e[xi]
    med_bost[i] = np.median(sample_bost)

sort_med_bost = np.sort(med_bost)

# fig , ax = plt.subplots(1)
# ax.hist(sort_med_bost,40)
# ax.axvline(sort_med_bost[841],color = 'r', linestyle='--')
# ax.axvline(sort_med_bost[158],color = 'r', linestyle='--')
# plt.show()

print("  (2) upper(15.9%) std = ", sort_med_bost[840]- sort_med_bost[500])
print("      lower(15.9%) std = ", sort_med_bost[499]- sort_med_bost[159],"\n")

e = e[(x<-10.6)]

med_bost2 = np.zeros(n)

for j in range(n):
    xj = np.random.randint(len(e),size = len(e))
    sample_bost = e[xj]
    med_bost2[j] = np.median(sample_bost)

sort_med_bost2 = np.sort(med_bost2)

print("  (3) e_mask_median = ",np.median(e))
print("      bootstrap std = ",sort_med_bost2[840]-sort_med_bost[500],sort_med_bost2[159]-sort_med_bost2[499])

#####2
n = 1000

def gx(r,theta,pi):
    f = lambda rr , pp : (r*np.cos(theta)*np.cos(pi)-rr*np.cos(pp))/(r**2+rr**2-2*r*rr*np.cos(theta)*np.cos(pp-pi))**(3/2)
    options={'limit':n}
    res, err = nquad(f, [[0, 2*np.pi], [0, 1]],opts=[options,options])
    return (1/np.pi)*res 

def gy(r,theta,pi):
    f = lambda rr , pp : (rr*np.sin(pp)- r*np.cos(theta)*np.sin(pi))/(r**2+rr**2-2*r*rr*np.cos(theta)*np.cos(pp-pi))**(3/2)
    options={'limit':n}
    res, err = nquad(f, [[0, 2*np.pi], [0, 1]],opts=[options,options])
    return (1/np.pi)*res

def gz(r,theta,pi):
    f = lambda rr , pp : (r*np.sin(theta))/(r**2+rr**2-2*r*rr*np.cos(theta)*np.cos(pp-pi))**(3/2)
    options={'limit':n}
    res, err = nquad(f, [[0, 2*np.pi], [0, 1]],opts=[options,options])
    return (1/np.pi)*res 

print(gy(0.1,0,0))

# def gx(r,theta,pi):
#     f = lambda rr , pp : (r*np.cos(theta)*np.cos(pi)-rr*np.cos(pp))/(r**2+rr**2-2*r*rr*np.cos(theta)*np.cos(pp-pi))**(3/2)
#     # return dblquad(f, 0, 2*np.pi, lambda pp:0, lambda pp:1)
#     res, err = dblquad(f, 0, 2*np.pi, lambda pp:0, lambda pp:1,limit = 100)
#     return (1/np.pi)*res

# def gy(r,theta,pi):
#     f = lambda rr , pp : (rr*np.sin(pp)- r*np.cos(theta)*np.sin(pi))/(r**2+rr**2-2*r*rr*np.cos(theta)*np.cos(pp-pi))**(3/2)
#     res, err = dblquad(f, 0, 2*np.pi, lambda pp:0, lambda pp:1,limit = 100)
#     return (1/np.pi)*res

# def gz(r,theta,pi):
#     f = lambda rr , pp : (r*np.sin(theta))/(r**2+rr**2-2*r*rr*np.cos(theta)*np.cos(pp-pi))**(3/2)
#     res, err = dblquad(f, 0, 2*np.pi, lambda pp:0, lambda pp:1, limit = 100)
#     return (1/np.pi)*res

# def g(r,theta,pi):
#     return np.sqrt(gx(r,theta,pi)**2 + gy(r,theta,pi)**2 + gz(r,theta,pi)**2)

# def g0(r):
#     return np.abs(2*(r/((1+r**2)**(1/2))-1))

# print(g(50,0,10))
# print(g0(50))


