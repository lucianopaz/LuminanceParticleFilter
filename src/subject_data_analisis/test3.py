#-*- coding: UTF-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from utils import normcdf
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
from scipy import optimize as opt

"""
Test the g = G(x) bijective relation when there are multiple variances with different probabilities
"""

class Foo:
	def __init__(self,v,p,v0,mu0):
		self.v = v.copy()
		self.p = p.copy()
		self.v0 = float(v0)
		self.mu0 = float(mu0)
	
	def _compat_views(self,*args):
		if not all([a.ndim==args[0].ndim for a in args]):
			raise RuntimeError("Input args must have the same number of dimensions")
		ndim = args[0].ndim
		args_ind = '['+','.join([':']*ndim)+',None]'
		args_view = [eval('a'+args_ind) for a in args]
		if len(args_view)==1:
			args_view = args_view[0]
		else:
			args_view = tuple(args_view)
		inner_ind = '['+','.join(['None']*ndim)+',:]'
		v,p = tuple([eval('i'+inner_ind) for i in [self.v,self.p]])
		return args_view,v,p
	
	def vt(self,t):
		t,v,p = self._compat_views(t)
		return (1/(t/v+1/self.v0))
	
	def mut(self,t,x):
		x,v,p = self._compat_views(x)
		return (x/v + self.mu0/self.v0)*self.vt(t)
	
	def x2g(self,t,x):
		(x_,t_),v,p = self._compat_views(x,t)
		_vt = self.vt(t)
		_st = np.sqrt(_vt)
		_mut = self.mut(t,x)
		return np.sum(p*_st*normcdf(_mut/_st),axis=-1)/np.sum(p*_st,axis=-1)
	
	def dx2g(self,t,x):
		(x_,t_),v,p = self._compat_views(x,t)
		_vt = self.vt(t)
		_st = np.sqrt(_vt)
		_mut = self.mut(t,x)
		return np.sum(p*_vt/v*np.exp(-0.5*_mut**2/_vt),axis=-1)/np.sum(p*_st,axis=-1)/np.sqrt(2*np.pi)

x = np.linspace(-20,20,1e5+1)[:,None]
t = np.linspace(0,10,5)[None,:]

v = (np.arange(4)+1)**2
p = np.ones_like(v)/float(len(v))
foo = Foo(v,p,1.,0.)

dx = x[1]-x[0]
g = foo.x2g(t,x)
dg = foo.dx2g(t,x)

numg = np.zeros_like(g)
for i in range(g.shape[1]):
	numg[:,i] = cumtrapz(dg[:,i],x[:,0],axis=0,initial=g[0,i])+g[0,i]
numdg = np.vstack(((g[1,:]-g[0,:])/dx,0.5*(g[2:,:]-g[:-2,:])/dx,(g[-1,:]-g[-2,:])/dx))

plt.figure()
plt.subplot(211)
lines = plt.plot(x,g)
for i,l in enumerate(lines):
	plt.plot(x[:,0],numg[:,i],'--',color=l.get_color())
plt.ylabel(r'$G(x)$')
plt.subplot(212)
lines = plt.plot(x,dg)
for i,l in enumerate(lines):
	plt.plot(x[:,0],numdg[:,i],'--',color=l.get_color())
plt.ylabel(r'$\frac{\partial G(x)}{\partial x}$')

plt.figure()
plt.subplot(211)
lines = plt.plot(x,g-numg)
plt.ylabel(r'$G(x)$ error')
plt.subplot(212)
lines = plt.plot(x,dg-numdg)
plt.ylabel(r'$\frac{\partial G(x)}{\partial x}$ error')

# Root finding

xts = np.array([-10,-0.7,3.,15.6842])
t = np.array([1.])
gts = foo.x2g(t,xts)
jac = lambda x: foo.dx2g(t,x)

for gt,xt in zip(gts,xts):
	f = lambda x: foo.x2g(t,x)-gt
	out = opt.root(f,0.,jac=jac,method='hybr')
	print(gt,xt,out.x,out.fun)

plt.show(True)
