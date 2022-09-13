import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m = 0.01

def dYdt( t, Y ) :                              # Y[0] = pi(t), Y[1] = u(t) = dpi(t)/dt
    return  Y[1], (-3)*np.sqrt(((Y[1]**2)/6)+(Y[0]**2)*(m**2)/6)*Y[1] - (m**2)*Y[0]   # pi'(t), u'(t)

Y0 = ( 9, 1) # pi(0)  u(0)

t0 = 0 # unit = sec??
tend = 1000

sol = solve_ivp( dYdt, (t0, tend), Y0, t_eval=np.linspace(t0,tend,101) )

H = np.sqrt(((sol.y[1]**2)/6)+(sol.y[0]**2)*(m**2)/6)
V = ((m**2)*(sol.y[0]**2))/2
dVdpi = m**2*sol.y[0]
slo_par = sol.y[1]**2/(2*(H**2))
slo_par_V = 2*(1/sol.y[0])**2


fig,ax = plt.subplots(2,3,figsize=(9,15))
ax[0][0].plot(sol.t,sol.y[0])
ax[0][0].set_xlabel("t")
ax[0][0].set_ylabel("$\phi$")
ax[0][1].plot(sol.t,sol.y[1])
ax[0][1].set_xlabel("t")
ax[0][1].set_ylabel("$\dot{\phi}$")
ax[0][2].plot(sol.y[0],sol.y[1])
ax[0][2].set_xlabel("$\phi$")
ax[0][2].set_ylabel("$\dot{\phi}$")
ax[1][0].plot(sol.t,slo_par)
ax[1][0].set_xlabel("t")
ax[1][0].set_ylabel("$\epsilon$")
ax[1][1].plot(sol.t,slo_par_V)
ax[1][1].set_xlabel("t")
ax[1][1].set_ylabel("$\epsilon_{V}$")
plt.show()