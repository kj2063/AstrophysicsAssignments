import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import NullFormatter
from astropy.cosmology import LambdaCDM
import corner
import emcee
n=10000
m_true = -0.9594
b_true = 4.294
f_true = 0.534  #log_f_true = -0.627

N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = m_true * x + b_true
y += np.abs(f_true * y) * np.random.randn(N)
y += yerr * np.random.randn(N)


def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

def metropolis_hastings(n):

    m0,b0,log_f0 = 0,0,0
    m_result = []
    b_result = []
    log_f_result = []

    for i in range(n):
        m_next, b_next, log_f_next = np.array([m0,b0,log_f0]) + np.random.uniform(-1,1,size=3)
    
        u = np.random.rand()
    
        if np.log(u) < log_probability((m_next,b_next,log_f_next), x=x, y=y, yerr=yerr)-log_probability((m0,b0,log_f0), x=x, y=y, yerr=yerr):
            m0, b0, log_f0 = m_next, b_next, log_f_next
            m_result.append(m0)
            b_result.append(b0)
            log_f_result.append(log_f0)

    return m_result, b_result, log_f_result

s = metropolis_hastings(n)

m_fin = s[0]
b_fin = s[1]
log_f_fin = s[2]

print("3. true_m = {}, true_b = {}, true_f = {}".format(m_true,b_true,np.log(f_true)))
print(np.mean(m_fin[20:]),np.mean(b_fin[20:]),np.mean(log_f_fin[20:]),"\n")



ndim, nwalkers = 3, 300
pos = [(m_fin[-1],b_fin[-1],log_f_fin[-1]) + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability ,threads=2)
sampler.run_mcmc(pos, 1000)
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
sampler.run_mcmc(pos, 1000)
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))

fig = corner.corner(samples,labels=["T", "Amp", r"$\beta$"],quantiles=[0.16, 0.5, 0.84],show_titles=True,truths=(m_fin[-1],b_fin[-1],log_f_fin[-1]))




# for_list = [[m_fin,b_fin],[m_fin,log_f_fin],[b_fin,log_f_fin]]
# list_name = [['m','b'],['m','log_f'],['b','log_f']]
# for l in zip(list_name,for_list):
#     x = l[1][0][10:]
#     y = l[1][1][10:]

#     nullfmt = NullFormatter()        

#     left, width = 0.1, 0.65
#     bottom, height = 0.1, 0.65
#     bottom_h = left_h = left + width + 0.02

#     rect_scatter = [left, bottom, width, height]
#     rect_histx = [left, bottom_h, width, 0.2]
#     rect_histy = [left_h, bottom, 0.2, height]

#     plt.figure(2, figsize=(8, 8))

#     axScatter = plt.axes(rect_scatter)
#     axHistx = plt.axes(rect_histx)
#     axHisty = plt.axes(rect_histy)

#     axHistx.xaxis.set_major_formatter(nullfmt)
#     axHisty.yaxis.set_major_formatter(nullfmt)

#     axScatter.scatter(x, y)

#     binwidth = 0.25
#     xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
#     lim = (int(xymax/binwidth) + 1) * binwidth

#     axScatter.set_xlabel(l[0][0])
#     axScatter.set_ylabel(l[0][1])

#     bins = np.arange(-lim, lim + binwidth, binwidth)
#     axHistx.hist(x, bins=20, density= True)
#     axHisty.hist(y, bins=20, orientation='horizontal', density= True)

#     axHistx.set_xlim(axScatter.get_xlim())
#     axHisty.set_ylim(axScatter.get_ylim())

#     plt.show()
