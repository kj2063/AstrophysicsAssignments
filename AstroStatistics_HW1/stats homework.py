import numpy as np
from scipy import stats

#3

theta = np.random.RandomState().uniform(1,100)

N = 100

rv = stats.pareto(b = theta+1, scale = 1).rvs(N)

print("\ntheta value :{}\n".format(theta))

#4

theta_hat_moment = 1/(rv.mean()-1)

print("theta_hat by method of the moments :{}\n".format(theta_hat_moment))

#5

sigma_lnx = 0

for i in rv:
    sigma_lnx += np.log(i)

theta_hat_MLE = (N/sigma_lnx) - 1

print("theta_hat by MLE :{}\n".format(theta_hat_MLE))

#6

bias = theta_hat_moment - theta

# Repeat the experiment N1=100 time

# rv = stats.pareto(b = theta+1, scale = 1).rvs(N*N1).reshape(N1,N)
# theta_hat_moment = 1/(rv.mean(axis = -1) - 1)
# bias = theta_hat_moment.mean() - theta

print("bias of theta_hat :{}\n".format(bias))

#7

N1 = 100

rv_X = stats.pareto(b = theta+1, scale = 1).rvs(N1*N).reshape(N1,N)

X_bar = np.mean(rv_X, axis = -1)

S = np.sqrt(rv_X.var(axis = -1))

alpha = 0.95
t1, t2 = stats.t(N-1).interval(alpha)


low = np.sqrt(N) / ((X_bar - 1)*np.sqrt(N) - t1*S)
high = np.sqrt(N) / ((X_bar - 1)*np.sqrt(N) - t2*S)

print("the number of theta in confidence interval(95%) :{}".format(((low<theta).__and__(theta<high)).sum()))






