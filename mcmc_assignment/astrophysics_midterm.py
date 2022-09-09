import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Q1

N = 100

x = np.random.uniform(1,10,N)

theta0 = np.random.uniform(-10,10)
theta1 = np.random.uniform(-10,10)

print("real value")
print("theta0 = {}".format(theta0))
print("theta1 = {}".format(theta1))

noise = np.random.normal(size = N)

y = theta0 + x*theta1 + noise

x0 = np.linspace(1,10)
y0 = theta0 + x0*theta1

one = np.ones_like(x)

XX = np.vstack((one, x)).T

theta_hat = np.linalg.solve( XX.T@XX , XX.T@y)

print("\nanalytical approach\ntheta0 = {}\ntheta1 = {}".format(theta_hat[0],theta_hat[1]))

fig, ax = plt.subplots(1,2)
ax[0].plot(x0,y0)
ax[0].plot(x,y,".")

#Q2

sigma_L = 1

def likelihood(theta0_p, theta1_p):
    return  (1 / ((2*np.pi*sigma_L**2)**(N/2)))*np.exp((-1/2)*np.sum(((y - theta0_p - theta1_p*x)/sigma_L)**2 ))

# mu=0 sigma = 1
def prior(theta0_p, theta1_p):
    return (1/(2*np.pi))*np.exp((-1/2)*(theta0_p**2 + theta1_p**2))

def posterior(theta0_p, theta1_p):
    return likelihood(theta0_p,theta1_p)*prior(theta0_p, theta1_p)

def metropolis_hastings(repeat):
    theta0_p, theta1_p = 0,0
    theta0_result = np.zeros(repeat)
    theta1_result = np.zeros(repeat)

    for i in range(repeat):
        theta0_pnext, theta1_pnext = np.array([theta0_p,theta1_p]) + np.random.uniform(-1,1,size=2)
        # if use too small sigma, only can jump small distance and that make hard to converge.
        # if use too big sigma, next theta0,1 might be not in the posterior probability, and that also make hard to converge 
        u = np.random.rand()
        # if posterior(theta0_pnext, theta1_pnext) > posterior(theta0_p, theta1_p):
        if u < min(1,posterior(theta0_pnext, theta1_pnext)/posterior(theta0_p, theta1_p)):
            theta0_p, theta1_p = theta0_pnext, theta1_pnext
        theta0_result[i] = theta0_p
        theta1_result[i] = theta1_p
        
    return theta0_result , theta1_result

s = metropolis_hastings(repeat= 10000)

print("\nMCMC approach\ntheta0 = {}\ntheta1 = {}".format(s[0][-1],s[1][-1]))
x = s[0]
y = s[1]

ax[1].plot(x, y, ".", markersize =2)

plt.show()