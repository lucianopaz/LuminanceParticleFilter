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
		ndim = None
		for a in args:
			if isinstance(a,np.ndarray):
				if ndim is None:
					ndim = a.ndim
				else:
					ndim = max([ndim,a.ndim])
		if ndim is None or ndim==0:
			if len(args)==1:
				return args,self.v,self.p
			else:
				return tuple(args),self.v,self.p
		else:
			args_view = []
			for a in args:
				if isinstance(a,np.ndarray):
					if a.ndim>0:
						args_ind = '['+','.join([':']*a.ndim)+','+','.join(['None']*(ndim+1-a.ndim))+']'
						args_view.append(eval('a'+args_ind))
					else:
						args_view.append(a)
				else:
					args_view.append(a)
			inner_ind = '['+','.join(['None']*ndim)+',:]'
			v,p = tuple([eval('i'+inner_ind) for i in [self.v,self.p]])
			if len(args)==1:
				return args_view[0],v,p
			else:
				return tuple(args_view),v,p
	
	def vt(self,t):
		t,v,p = self._compat_views(np.array(t))
		return (1/(t/v+1/self.v0))
	
	def mut(self,t,x):
		x,v,p = self._compat_views(np.array(x))
		return (x/v + self.mu0/self.v0)*self.vt(t)
	
	def x2g(self,t,x):
		(x_,t_),v,p = self._compat_views(np.array(x),np.array(t))
		_vt = self.vt(t)
		_st = np.sqrt(_vt)
		_mut = self.mut(t,x)
		return np.sum(p*_st*normcdf(_mut/_st),axis=-1)/np.sum(p*_st,axis=-1)
	
	def dx2g(self,t,x):
		(x_,t_),v,p = self._compat_views(np.array(x),np.array(t))
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
	numg[:,i] = cumtrapz(dg[:,i],x[:,0],axis=0,initial=0)+g[0,i]
numdg = np.vstack(((g[1,:]-g[0,:])/dx,0.5*(g[2:,:]-g[:-2,:])/dx,(g[-1,:]-g[-2,:])/dx))

print(np.max(np.abs(g-numg)),np.max(np.abs(dg-numdg)))

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

xts = np.linspace(-10,10,1001)
t = np.array([1.])
gts = foo.x2g(t,xts)
jac = lambda x: foo.dx2g(t,x)

import time

start_hybr = time.clock()
hybr_error = np.zeros_like(xts)
for i,gt in enumerate(gts):
	f = lambda x: foo.x2g(t,x)-gt
	if i>0:
		out = opt.root(f,out.x,jac=jac,method='hybr')
	else:
		out = opt.root(f,0.,jac=jac,method='hybr')
	hybr_error[i] = np.abs(xts[i]-out.x)
end_hybr = time.clock()

start_lm = time.clock()
lm_error = np.zeros_like(xts)
for i,gt in enumerate(gts):
	f = lambda x: foo.x2g(t,x)-gt
	if i>0:
		out = opt.root(f,out.x,jac=jac,method='lm')
	else:
		out = opt.root(f,0.,jac=jac,method='lm')
	lm_error[i] = np.abs(xts[i]-out.x)
end_lm = time.clock()

start_newton = time.clock()
newton_error = np.zeros_like(xts)
for i,gt in enumerate(gts):
	f = lambda x: foo.x2g(t,x)-gt
	if i>0:
		out = opt.newton(f, out, fprime=jac)
	else:
		out = opt.newton(f, 0., fprime=jac)
	newton_error[i] = np.abs(xts[i]-out)
end_newton = time.clock()

print('Hybr')
print(end_hybr-start_hybr,np.max(hybr_error))
print('Lm')
print(end_lm-start_lm,np.max(lm_error))
print('Newton')
print(end_newton-start_newton,np.max(newton_error))

plt.show(True)
