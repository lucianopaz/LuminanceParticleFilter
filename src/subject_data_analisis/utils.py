#!/usr/bin/python
#-*- coding: UTF-8 -*-

from __future__ import division
import numpy as np
from scipy import optimize
from scipy import stats
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

def average_downsample(a,output_len,axis=None,ignore_nans=True,dtype=np.float):
	"""
	b = average_downsample(a,output_len,axis=None,ignore_nans=True,dtype=np.float)
	
	This function downsamples a numpy array of arbitrary shape. It does
	so by computing the average value of the input array 'a' inside a
	window of steps in order to produce an array with the supplied
	output_len. The output_len does not need to be submultiple of the
	original array's shape, and the function handles the averaging of
	the edges of each window properly.
	
	Inputs:
	- a:           np.array that will be downsampled
	- output_len:  Scalar that specifies the length of the output in the
	               downsampled axis.
	- axis:        The axis of the input array along which the array
	               will be downsampled. By default axis is None, and in
	               that case, the downsampling is performed on the
	               flattened array.
	- ignore_nans: Bool that specifies whether to ignore nans in while
	               averaging or not. Default is to ignore.
	- dtype:       Specifies the output array's dtype. Default is
	               np.float
	
	Output:
	- b:           The downsampled array. If axis is None, it will be a
	               flat array of shape equal to (int(output_len)).
	               If axis is not None, 'b' will have the same shape as
	               'a' except for the specified axis, that will have
	               int(output_len) elements
	
	Example
	>>> import numpy as np
	>>> a = a = np.reshape(np.arange(100,dtype=np.float),(10,-1))
	>>> a[-1] = np.nan
	
	>>> utils.average_downsample(a,10)
	array([  4.5,  14.5,  24.5,  34.5,  44.5,  54.5,  64.5,  74.5,  84.5,   nan])
	
	>>> utils.average_downsample(a,10,axis=0)
	array([[ 40.,  41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.,  49.]])
	
	>>> utils.average_downsample(a,10,axis=1)
	array([[  4.5],
	       [ 14.5],
	       [ 24.5],
	       [ 34.5],
	       [ 44.5],
	       [ 54.5],
	       [ 64.5],
	       [ 74.5],
	       [ 84.5],
	       [  nan]])
	
	"""
	if axis is None:
		a = a.flatten()
		axis = 0
		sum_weight = 0
		window = float(a.shape[0])/float(output_len)
		b = np.zeros((int(output_len)),dtype=dtype)
	else:
		a = np.swapaxes(a,0,axis)
		sum_weight = np.zeros_like(a[0],dtype=np.float)
		b_shape = list(a.shape)
		window = float(b_shape[0]/output_len)
		b_shape[0] = int(output_len)
		b = np.zeros(tuple(b_shape),dtype=dtype)
	flat_array = a.ndim==1
	
	step_size = 1./window
	position = 0.
	i = 0
	prev_index = 0
	L = len(a)
	Lb = len(b)
	all_indeces = np.ones_like(a[0],dtype=np.bool)
	step = True
	while step:
		if ignore_nans:
			valid_indeces = np.logical_not(np.isnan(a[i]))
		else:
			valid_indeces = all_indeces
		position = (i+1)*step_size
		index = int(position)
		if prev_index==index:
			weight = valid_indeces.astype(dtype)*step_size
			sum_weight+= weight
			if flat_array:
				b[index]+= a[i]*weight if valid_indeces else 0.
			else:
				b[index][valid_indeces]+= a[i][valid_indeces]*weight[valid_indeces]
		elif prev_index!=index:
			weight = valid_indeces*(position-index)
			prev_weight = valid_indeces*(index+step_size-position)
			if flat_array:
				b[prev_index]+= a[i]*prev_weight if valid_indeces else 0.
				sum_weight+= prev_weight
				b[prev_index]/=sum_weight
				if index<Lb:
					b[index]+= a[i]*weight if valid_indeces else 0.
			else:
				b[prev_index][valid_indeces]+= a[i][valid_indeces]*prev_weight[valid_indeces]
				sum_weight+= prev_weight
				b[prev_index]/=sum_weight
				if index<Lb:
					b[index][valid_indeces]+= a[i][valid_indeces]*weight[valid_indeces]
			sum_weight = weight
		
		prev_index = index
		
		i+=1
		if i==L:
			step = False
			if index<Lb:
				b[index]/=sum_weight
	b = np.swapaxes(b,0,axis)
	return b

def holm_bonferroni(p_in,alpha=None):
	"""
	holm_bonferroni(p_in,alpha=None)
	Correct the p values for multiple tests using holm-bonferroni criteria

	Inputs:
	 p_in:  a 1D numpy array with the original p values
	 alpha: Optional input. Default is None. If not None, it must be a float
			that specifies the admisibility for the test given by the
			corrected p values.

	Outputs:
	 p2: The corrected p values
	 h:  If alpha is not None, the output is a tuple formed by (p2,h) where
		 h is an array of bools that is true or false indicating if the
		 corrected p value is lower than the supplied alpha or not
	"""
	p = p_in.flatten()
	sort_inds = np.argsort(p)
	reverse_sort_inds = np.zeros_like(sort_inds)
	for i,si in enumerate(sort_inds):
		reverse_sort_inds[si] = i
	p = p[sort_inds]
	
	p2 = np.zeros_like(p)
	m = np.sum(np.logical_not(np.isnan(p)).astype(np.int))
	for i in range(m):
		temp = (m-np.arange(i+1))*p[:i+1]
		temp[temp>1] = 1.
		p2[i] = np.max(temp)
	p2 = p2[reverse_sort_inds]
	if alpha:
		alpha = float(alpha)
		comparison = p > (alpha/(m-np.arange(m)));
		if np.all(comparison):
			h = np.zeros_like(p,dtype=np.bool)
		else:
			h = np.ones_like(p,dtype=np.bool)
			k = comparison.nonzero()[0][0]
			h[k:] = False
		return p2,h
	else:
		return p2

def corrcoef(a,b=None,axis=0,method='pearson',nanhandling='pairwise'):
	"""
	corrcoef(a,b=None,axis=0,method='pearson',nanhandling='pairwise')
	
	Input:
	 a:           Mandatory input array. Can be a 1D or 2D numpy ndarray.
	              The different dimensions of 'a' represent the observed
	              data sample, and each of the variables. This representation
	              depends on the supplied axis
	 b:           Optional input array. If 'b' is None, it is ignored. If
	              it is not None, 'b' is treated as an additional variable
	              of the array 'a'.
	 axis:        Default is 0. axis determines which dimension holds the
	              data samples. If axis is 0, the rows are assumed to hold
	              separate data samples, and the columns are assumed to
	              represent different variables associated to each sample.
	              If axis is 1, this assumption is transposed. If axis is
	              None, 'a' and 'b' are flattened before computing the
	              correlation coeficient.
	 method:      A string indicating the method that will be used to
	              compute the correlation. Can be 'pearson', 'spearman'
	              or 'kendall'. Refer to the functions pearsonr, spearmanr
	              and kendalltau defined in scipy.stats for further
	              details
	 nanhandling: Determines how nans are handled while computing the
	              correlation coeficients. Can be None, 'pairwise' or
	              'complete'. If None, no special handling is done before
	              attempting to compute the correlation coefficients.
	              If 'complete', all the data observations that hold at least
	              one variable which is nan are discarded before computing
	              the correlation coefficients. If 'pairwise', rho[i,j] is
	              computed using rows with no NaN values in the variables
	              i or j.
	"""
	if axis is None:
		a = a.flatten()
		if not b is None:
			b = b.flatten()
		axis=0
	if a.ndim>2:
		raise ValueError("a must be a 1D or 2D numpy array")
	if not b is None:
		if b.ndim>2:
			raise ValueError("b must be None or a 1D or 2D numpy array")
		if axis==0:
			a = np.hstack((a,b))
		else:
			a = np.vstack((a,b)).T
	try:
		calculator = {'pearson':stats.pearsonr,'spearman':stats.spearmanr,'kendall':stats.kendalltau}[str(method).lower()]
	except:
		raise ValueError('Supplied method must can be "pearson", "spearman" or "kendall". Got {0} instead'.format(method))
	
	rho = np.zeros((len(a),len(a)))
	pval = np.zeros((len(a),len(a)))
	if nanhandling is None:
		for i,ai in enumerate(a):
			rho[i,i],pval[i,i] = calculator(ai,ai)
			for j,aj in enumerate(a[i+1:]):
				rho[j+i+1,i],pval[j+i+1,i] = rho[i,j+i+1],pval[i,j+i+1] = calculator(ai,aj)
	else:
		nanhandling = str(nanhandling).lower()
		if nanhandling=='pairwise':
			for i,ai in enumerate(a):
				valid = np.logical_not(np.isnan(ai))
				rho[i,i],pval[i,i] = calculator(ai[valid],ai[valid])
				for j,aj in enumerate(a[i+1:]):
					valid = np.logical_not(np.logical_or(np.isnan(ai),np.isnan(aj)))
					rho[j+i+1,i],pval[j+i+1,i] = rho[i,j+i+1],pval[i,j+i+1] = calculator(ai[valid],aj[valid])
		elif nanhandling=='complete':
			valid = np.logical_not(np.any(np.isnan(a),axis=0))
			for i,ai in enumerate(a):
				rho[i,i],pval[i,i] = calculator(ai,ai)
				for j,aj in enumerate(a[i+1:]):
					rho[j+i+1,i],pval[j+i+1,i] = rho[i,j+i+1],pval[i,j+i+1] = calculator(ai[valid],aj[valid])
		else:
			raise ValueError("nanhandling must be None, 'complete' or 'pairwise'. Got {0} instead.".format(nanhandling))
	return rho,pval
