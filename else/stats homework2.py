import numpy as np
from scipy import stats, special, optimize
import matplotlib.pyplot as plt

k = np.random.randint(1,30)
df = 2*k
theta = 2

print("\nvalue of k :{}".format(k))
print("value of theta:{}\n".format(2))

#4

N = 1000
NN = 1000


x = stats.chi2.rvs(df,size = N*NN).reshape(NN,N)

#5

theta_hat = ((x**2).mean(axis=-1) - x.mean(axis=-1)**2) / (x.mean(axis=-1))
k_hat = x.mean(axis=-1)**2 / ((x**2).mean(axis=-1) - x.mean(axis=-1)**2)

print("value(from MoM) of k_hat :{}".format(k_hat[0]))
print("value(from MoM) of theta_hat :{}\n".format(theta_hat[0]))

#6

sig_lnx = np.log(x).sum(axis=-1)

sol_kk = []

for i in range(NN):
    sig_lnx_i = np.log(x[i]).sum()
    x_i = x[i]
    def dF_k(kk):
        return -(sig_lnx_i - N*np.log(x_i.mean()/kk) - N*special.digamma(kk))
    sol = optimize.root_scalar(dF_k, bracket =[0.1,100], method='bisect')
    sol_kk.append(sol.root)
    
sol_kk = np.array(sol_kk)

the = x.mean(axis=-1)/sol_kk

print("value(from MLE) of k_hat :{}".format(sol_kk[0]))
print("value(from MLE) of theta_hat :{}\n".format(the[0]))

#7

fig, ax = plt.subplots(2,2)
ax[0][0].hist(k_hat, density=True)
ax[0][1].hist(theta_hat, density=True)
ax[1][0].hist(sol_kk, density=True)
ax[1][1].hist(the, density=True)

#8

bias_kMoM = k_hat.mean() - k
bias_kMLE = sol_kk.mean() - k
bias_theMoM = theta_hat.mean() - theta
bias_theMLE = the.mean() - theta

var_kMoM = (k_hat**2).mean() - (k_hat.mean())**2
var_kMLE = (sol_kk**2).mean() - (sol_kk.mean())**2
var_theMoM = (theta_hat**2).mean() - (theta_hat.mean())**2
var_theMLE = (the**2).mean() - (the.mean())**2

risk_kMoM = var_kMoM + bias_kMoM**2
risk_kMLE = var_kMLE + bias_kMLE**2
risk_theMoM = var_theMoM + bias_theMoM**2
risk_theMLE = var_theMLE + bias_theMLE**2
print("                 k    theta")
print("risk of MoM {:6f} {:6f}".format(risk_kMoM,risk_theMoM),"\nrisk of MLE {:6f} {:6f}".format(risk_kMLE,risk_theMLE))

