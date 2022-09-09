import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import NullFormatter
from astropy.cosmology import LambdaCDM

###1,2

n = 50000 # the number of trying mcmc

x1 = 0 

def target(x1):
    return stats.norm.pdf(x1,1,1)

result = np.zeros(n)

for i in range(n):
    x_next = x1 + np.random.uniform(-1,1)

    r = np.random.uniform(0,1)
    if r < (target(x_next)/target(x1)):
        x1 = x_next
    result[i] = x1

x0 = np.linspace(-5,5)

fig, ax = plt.subplots(1)
ax.hist(result,40,density =True)
ax.plot(x0,stats.norm.pdf(x0,1,1))
ax.set_xlim(-5,5)
plt.show()


###3

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

###4

fig, ax = plt.subplots(1)
ax.plot(m_fin,color = 'b' , label = "m")
ax.hlines(y=m_true, xmin=0, xmax=100 ,color='b' )
ax.plot(b_fin,color = 'g',label = "b")
ax.hlines(y=b_true, xmin=0, xmax=100,color = 'g')
ax.plot(log_f_fin, color = 'r' ,label = "log_f")
ax.hlines(y=np.log(f_true), xmin=0, xmax=100 ,color ="r")
ax.legend()
ax.set_xlim(0,100)
plt.show()

#reasonable threshold for burn-in = about 10 times jumps , I think, It can be changed by starting point

###5
alpha = 0.68

m_t1, m_t2 = stats.norm(loc =np.mean(m_fin),scale = np.std(m_fin)).interval(alpha)
b_t1, b_t2 = stats.norm(loc =np.mean(b_fin),scale = np.std(b_fin)).interval(alpha)
log_f_t1, log_f_t2 = stats.norm(loc =np.mean(log_f_fin),scale = np.std(log_f_fin)).interval(alpha)

print("5.68% confidence intervals of m = {} {}".format(m_t1,m_t2))
print("  68% confidence intervals of b = {} {}".format(b_t1,b_t2))
print("  68% confidence intervals of log_f = {} {}\n".format(log_f_t1,log_f_t2))

###6

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

def log_lik(theta, zcmb, mb, dmb):
    Om, Ol, MB = theta
    model = mucdm(zcmb,Om,Ol,MB)
    return (-1/2)*np.sum(((mb - model) / dmb)**2)

def log_pri(theta):
    Om, Ol, MB = theta
    if 0 < Om < 1 and 0.0 < Ol < 1 and -30 < MB < 0:
        return 0.0
    return -np.inf

def log_prob(theta, zcmb, mb, dmb):
    lp = log_pri(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_lik(theta, zcmb, mb, dmb)

def metropolis_hastings(n):

    Om0, Ol0, MB0 = 0,0,0
    Om_result = []
    Ol_result = []
    MB_result = []

    for i in range(n):
        Om_next, Ol_next, MB_next = np.array([Om0,Ol0,MB0]) + np.random.uniform(-1,1,size=3)
    
        u = np.random.rand()
    
        if np.log(u) < log_prob((Om_next, Ol_next, MB_next), zcmb, mb, dmb)-log_prob((Om0, Ol0, MB0),zcmb, mb, dmb):
            Om0, Ol0, MB0 = Om_next, Ol_next, MB_next
            Om_result.append(Om0)
            Ol_result.append(Ol0)
            MB_result.append(MB0)

    return Om_result, Ol_result, MB_result

ss = metropolis_hastings(50000)


Om_fin = ss[0]
Ol_fin = ss[1]
MB_fin = ss[2]

print("6.Om =", np.mean(Om_fin[30:]),"\n  Ol =",np.mean(Ol_fin[30:]),"\n  MB =" ,np.mean(MB_fin[30:]))

for_list = [[Om_fin,Ol_fin],[Om_fin,MB_fin],[Ol_fin,MB_fin]]
list_name = [['Om','Ol'],['Om','MB'],['Ol','MB']]
for l in zip(list_name,for_list):
    x = l[1][0][10:]
    y = l[1][1][10:]

    nullfmt = NullFormatter()        

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    plt.figure(2, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axScatter.scatter(x, y)

    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlabel(l[0][0])
    axScatter.set_ylabel(l[0][1])

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=20, density= True)
    axHisty.hist(y, bins=20, orientation='horizontal', density= True)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()

###7


# def log_prob(theta, zcmb, mb, dmb):
#     lp = log_pri(theta)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp + log_lik(theta, zcmb, mb, dmb)

ndim = 3

np.random.seed(42)
means = np.random.rand(ndim)

cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov, cov)