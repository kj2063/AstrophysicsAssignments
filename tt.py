import pandas as pd
import ssl
import numpy as np
import matplotlib.pyplot as plt

print (pd.__version__)

# Read the data: 
def read_file(myfile):
  filein = "http://astrostatistics.psu.edu/MSMA/datasets/{}_profile.dat".format(myfile)
  ssl._create_default_https_context = ssl._create_unverified_context
  gal = pd.read_csv(filein,delim_whitespace=True)
  # NGC4472.describe()
  return gal
NGC4472 = read_file("NGC4472")
NGC4406 = read_file("NGC4406")
NGC4551 = read_file("NGC4551")

def mu(rr,AA,nn,re):
  bn = -0.868*nn+0.142
  return AA -2.5 * bn * ((rr/re)**(1/nn)-1)

  fig,ax = plt.subplots(1)

rr = NGC4472['radius'].values
mui = NGC4472['surf_mag'].values
ax.plot(rr,mui,ls='',marker='.')


ax.legend()


def chisqplot(xi, xf,nx, yi, yf,ny, xname, yname, ind, val):
  '''Plot
  Input: 
    xi: 
    yi: 
    nx: 
    yi:
    yf: 
    ny:
    xname:
    yname:
    ind:
    val:
  '''
  xs = np.linspace(xi,xf,nx)
  ys = np.linspace(yi,yf,ny)
  if ind==0: 
       chisqq = ((mui-mu(rr,AA=val,nn=xs[:,np.newaxis,np.newaxis],re=ys[np.newaxis,:,np.newaxis]))**2).sum(axis=-1)
  elif ind==1: 
      chisqq = ((mui-mu(rr,xs[:,np.newaxis,np.newaxis],nn=val,re=ys[np.newaxis,:,np.newaxis]))**2).sum(axis=-1)
  elif ind==2: 
      chisqq = ((mui-mu(rr,AA=xs[:,np.newaxis,np.newaxis],nn=ys[np.newaxis,:,np.newaxis],re=val))**2).sum(axis=-1)


  from matplotlib import cm
  cmap = cm.get_cmap('Blues_r', 11)

  fig, ax = plt.subplots()
  img = ax.imshow(np.log(chisqq), interpolation='bilinear', cmap=cmap, extent=[xi,xf,yf,yi], aspect='auto')

  ax.invert_yaxis()
  fig.colorbar(img, ax=ax)
  ax.set_xlabel(xname)
  ax.set_ylabel(yname)

chisqplot(2,10,101,100,300,121,"$n$","$r_e$",0,23)