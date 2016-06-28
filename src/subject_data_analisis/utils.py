#!/usr/bin/python
#-*- coding: UTF-8 -*-

from __future__ import division
import numpy as np
from scipy import optimize
import math


a = np.array([ 0.886226899, -1.645349621,  0.914624893, -0.140543331])
b = np.array([-2.118377725,  1.442710462, -0.329097515,  0.012229801])
c = np.array([-1.970840454, -1.624906493,  3.429567803,  1.641345311])
d = np.array([ 3.543889200,  1.637067800])
y0 = 0.7

def erfinv(y):
	if y<-1. or y>1.:
		raise ValueError("erfinv(y) argument out of range [-1.,1]")
	if abs(y)==1.:
		# Precision limit of erf function
		x = y*5.9215871957945083
	elif y<-y0:
		z = math.sqrt(-math.log(0.5*(1.0+y)))
		x = -(((c[3]*z+c[2])*z+c[1])*z+c[0])/((d[1]*z+d[0])*z+1.0)
	else:
		if y<y0:
			z = y*y;
			x = y*(((a[3]*z+a[2])*z+a[1])*z+a[0])/((((b[3]*z+b[2])*z+b[1])*z+b[0])*z+1.0)
		else:
			z = np.sqrt(-math.log(0.5*(1.0-y)))
			x = (((c[3]*z+c[2])*z+c[1])*z+c[0])/((d[1]*z+d[0])*z+1.0)
		# Polish to full accuracy
		x-= (math.erf(x) - y) / (2.0/math.sqrt(math.pi) * math.exp(-x*x));
		x-= (math.erf(x) - y) / (2.0/math.sqrt(math.pi) * math.exp(-x*x));
	return x

_vectErf = np.vectorize(math.erf,otypes=[np.float])
_vectErfinv = np.vectorize(erfinv,otypes=[np.float])
_vectGamma = np.vectorize(math.gamma,otypes=[np.float])

def normpdf(x, mu=0., sigma=1.):
	u = (x-mu)/sigma
	return 0.3989422804014327/np.abs(sigma)*np.exp(-0.5*u*u)

def normcdf(x,mu=0.,sigma=1.):
	"""
	Compute normal cummulative distribution with mean mu and standard
	deviation sigma. x, mu and sigma can be a numpy arrays that broadcast
	together.
	
	Syntax:
	y = normcdf(x,mu=0.,sigma=1.)
	"""
	try:
		new_x = (x-mu)/sigma
	except ZeroDivisionError:
		new_x = np.sign(x-mu)*np.inf
	return 0.5 + 0.5*_vectErf(new_x / np.sqrt(2.0))

def normcdfinv(y,mu=0.,sigma=1.):
	"""
	Compute the inverse of the normal cummulative distribution with mean
	mu and standard deviation sigma. y, mu and sigma can be a numpy
	arrays that broadcast together.
	
	Syntax:
	x = normcdfinv(y,mu=0.,sigma=1.)
	"""
	x = np.sqrt(2.0)*_vectErfinv(2.*(y-0.5))
	try:
		iterator = iter(sigma)
	except TypeError:
		if sigma==0.:
			raise ValueError("Invalid sigma supplied to normcdfinv. sigma cannot be 0")
		x = sigma*x+mu
	else:
		if any(sigma==0):
			raise ValueError("Invalid sigma supplied to normcdfinv. sigma cannot be 0")
		x = sigma*x+mu
	return x

def normgamma(x,t,mu=0.,l=1.,beta=2.,alpha=2.):
	return beta**alpha/_vectGamma(alpha)*np.sqrt(0.5*l/np.pi)*t**(alpha-0.5)*np.exp(-0.5*l*t*(x-mu)**2-beta*t)

def norminvgamma(x,sigma,mu=0.,l=1.,beta=2.,alpha=2.):
	return normgamma(x,sigma**(-2),mu,l,beta,alpha)

def average_downsample(a,ratio,axis=None,ignore_nans=True):
	#~ if ratio%1==0:
		#~ if ignore_nans:
			#~ mean = np.nanmean
		#~ else:
			#~ mean = np.mean
		#~ if axis is None:
			#~ b = mean(np.reshape(a.flatten(),(-1,ratio)),axis=1)
		#~ else:
			#~ new_shape = list(a.shape)
			#~ new_shape.insert(axis+1,ratio)
			#~ new_shape[axis] = -1
			#~ b = mean(np.reshape(a,tuple(new_shape)),axis=axis+1)
	#~ else:
	if axis is None:
		a = a.flatten()
		axis = 0
		sum_weight = 0
		b = np.zeros((int(np.ceil(a.shape[0]/ratio))))
	else:
		a = np.swapaxes(a,0,axis)
		sum_weight = np.zeros_like(a[0])
		b_shape = list(a.shape)
		b_shape[0] = int(np.ceil(b_shape[0]/ratio))
		b = np.zeros(tuple(b_shape))
	flat_array = a.ndim==1
	
	step_size = 1./ratio
	position = 0.
	i = 0
	prev_index = 0
	L = len(a)
	Lb = len(b)
	all_indeces = np.ones_like(a[0],dtype=np.bool)
	step = True
	print a.shape, b.shape
	while step:
		if ignore_nans:
			valid_indeces = np.logical_not(np.isnan(a[i]))
		else:
			valid_indeces = all_indeces
		position = (i+1)*step_size
		index = int(position)
		if prev_index==index:
			weight = valid_indeces*step_size
			sum_weight+= weight
			if flat_array:
				b[index]+= a[i]*weight if valid_indeces else 0.
			else:
				b[index][valid_indeces]+= a[i][valid_indeces]*weight
		elif prev_index!=index:
			weight = position-index
			prev_weight = index+step_size-position
			if flat_array:
				b[prev_index]+= a[i]*prev_weight if valid_indeces else 0.
				sum_weight+= prev_weight
				b[prev_index]/=sum_weight
				if index<Lb:
					b[index]+= a[i]*weight if valid_indeces else 0.
			else:
				b[prev_index][valid_indeces]+= a[i][valid_indeces]*prev_weight
				sum_weight+= prev_weight
				b[prev_index]/=sum_weight
				if index<Lb:
					b[index][valid_indeces]+= a[i][valid_indeces]*weight
			sum_weight = weight
		
		prev_index = index
		
		i+=1
		if i==L:
			step = False
			if index<Lb:
				b[index]/=sum_weight
	b = np.swapaxes(b,0,axis)
	return b
