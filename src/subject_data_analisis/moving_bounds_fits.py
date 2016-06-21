from __future__ import division

import enum, os, sys, math, scipy, pickle, cma
import data_io as io
import cost_time as ct
import numpy as np
from utils import normpdf
import matplotlib as mt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if np.__version__<'1.9':
	orig_unique = np.unique
	def np19_unique(ar, return_index=False, return_inverse=False, return_counts=False):
		"\n    Find the unique elements of an array.\n\n    Returns the sorted unique elements of an array. There are three optional\n    outputs in addition to the unique elements: the indices of the input array\n    that give the unique values, the indices of the unique array that\n    reconstruct the input array, and the number of times each unique value\n    comes up in the input array.\n\n    Parameters\n    ----------\n    ar : array_like\n        Input array. This will be flattened if it is not already 1-D.\n    return_index : bool, optional\n        If True, also return the indices of `ar` that result in the unique\n        array.\n    return_inverse : bool, optional\n        If True, also return the indices of the unique array that can be used\n        to reconstruct `ar`.\n    return_counts : bool, optional\n        .. versionadded:: 1.9.0\n        If True, also return the number of times each unique value comes up\n        in `ar`.\n\n    Returns\n    -------\n    unique : ndarray\n        The sorted unique values.\n    unique_indices : ndarray, optional\n        The indices of the first occurrences of the unique values in the\n        (flattened) original array. Only provided if `return_index` is True.\n    unique_inverse : ndarray, optional\n        The indices to reconstruct the (flattened) original array from the\n        unique array. Only provided if `return_inverse` is True.\n    unique_counts : ndarray, optional\n        .. versionadded:: 1.9.0\n        The number of times each of the unique values comes up in the\n        original array. Only provided if `return_counts` is True.\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Examples\n    --------\n    >>> np.unique([1, 1, 2, 2, 3, 3])\n    array([1, 2, 3])\n    >>> a = np.array([[1, 1], [2, 3]])\n    >>> np.unique(a)\n    array([1, 2, 3])\n\n    Return the indices of the original array that give the unique values:\n\n    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])\n    >>> u, indices = np.unique(a, return_index=True)\n    >>> u\n    array(['a', 'b', 'c'],\n           dtype='|S1')\n    >>> indices\n    array([0, 1, 3])\n    >>> a[indices]\n    array(['a', 'b', 'c'],\n           dtype='|S1')\n\n    Reconstruct the input array from the unique values:\n\n    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])\n    >>> u, indices = np.unique(a, return_inverse=True)\n    >>> u\n    array([1, 2, 3, 4, 6])\n    >>> indices\n    array([0, 1, 4, 3, 1, 2, 1])\n    >>> u[indices]\n    array([1, 2, 6, 4, 2, 3, 2])\n\n    "
		_return_inverse = return_inverse
		if return_counts:
			_return_inverse = True
		ret = orig_unique(ar,return_index=return_index,return_inverse=_return_inverse)
		if return_counts:
			if return_index:
					unique_inverse = ret[2]
			else:
				unique_inverse = ret[1]
			unique_counts = np.zeros_like(ret[0])
			for i,u in enumerate(ret[0]):
					unique_counts[i] = np.sum(ar==u)
			ret+=(unique_counts,)
		return ret
	np.unique = np19_unique

if np.__version__<'1.9':
	def np18_nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
		"Compute the arithmetic mean along the specified axis, ignoring NaNs.\n\nReturns the average of the array elements.  The average is taken over the flattened array by default, otherwise over the specified axis. `float64` intermediate and return values are used for integer inputs.\n\nFor all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.\n.. versionadded:: 1.8.0\n\nParameters\n----------\na : array_like\n    Array containing numbers whose mean is desired. If `a` is not an array, a conversion is attempted.\naxis : int, optional\n    Axis along which the means are computed. The default is to compute the mean of the flattened array.\ndtype : data-type, optional\n    Type to use in computing the mean.  For integer inputs, the default is `float64`; for inexact inputs, it is the same as the input dtype.\nout : ndarray, optional\n    Alternate output array in which to place the result.  The default is ``None``; if provided, it must have the same shape as the expected output, but the type will be cast if necessary.  See `doc.ufuncs` for details.\nkeepdims : bool, optional\n    If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the original `arr`.\n\nReturns\n-------\nm : ndarray, see dtype parameter above\n    If `out=None`, returns a new array containing the mean values, otherwise a reference to the output array is returned. Nan is returned for slices that contain only NaNs.\n\nSee Also\n--------\naverage : Weighted averagemean : Arithmetic mean taken while not ignoring NaNs\nvar, nanvar\n\nNotes\n-----\nThe arithmetic mean is the sum of the non-NaN elements along the axis divided by the number of non-NaN elements.\n\nNote that for floating-point input, the mean is computed using the same precision the input has.  Depending on the input data, this can cause the results to be inaccurate, especially for `float32`.  Specifying a higher-precision accumulator using the `dtype` keyword can alleviate this issue.\n\nExamples\n--------\n>>> a = np.array([[1, np.nan], [3, 4]])\n>>> np.nanmean(a)\n2.6666666666666665\n>>> np.nanmean(a, axis=0)\narray([ 2.,  4.])\n>>> np.nanmean(a, axis=1)\narray([ 1.,  3.5])"
		nans = np.isnan(a)
		n = np.sum(np.logical_not(nans),axis=axis,dtype=dtype,keepdims=keepdims);
		b = a[:]
		b[nans] = 0
		if out is None:
			out = np.sum(b,axis=axis,dtype=dtype,keepdims=keepdims)
		else:
			np.sum(b,axis=axis,dtype=dtype,out=out,keepdims=keepdims)
		out/=n
		return out
	np.nanmean = np18_nanmean

class Location(enum.Enum):
	facu = 0
	home = 1
	cluster = 2
	unknown = 3

opsys,computer_name,kern,bla,bits = os.uname()
if opsys.lower().startswith("linux"):
	if computer_name=="facultad":
		loc = Location.facu
	elif computer_name.startswith("sge"):
		loc = Location.cluster
elif opsys.lower().startswith("darwin"):
	loc = Location.home
else:
	loc = Location.unknown

if loc==Location.facu:
	data_dir='/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles'
elif loc==Location.home:
	data_dir='/Users/luciano/Facultad/datos'
elif loc==Location.cluster:
	data_dir='/homedtic/lpaz/DecisionConfidenceKernels/data'
elif loc==Location.unknown:
	raise ValueError("Unknown data_dir location")

time_units = 'seconds'
ISI = 0.04 #seconds
distractor = 50. # cd/m^2
patch_sigma = 5. # cd/m^2
model_var = (patch_sigma**2)*2/ISI
def set_time_units(units='seconds'):
	if units not in ('seconds','milliseconds'):
		raise ValueError("Invalid time units '{0}'. Available units are seconds and milliseconds".format(units))
	global time_units
	global ISI
	global model_var
	time_units = units
	if time_units=='seconds':
		ISI = 0.04
		model_var = (patch_sigma**2)*2/ISI
	else:
		ISI = 40.
		model_var = (patch_sigma**2)*2/ISI

def add_dead_time(gs,dt,dead_time,dead_time_sigma,mode='full'):
	"""
	new_gs = add_dead_time(gs,dt,dead_time_sigma,mode='full')
	"""
	if dead_time_sigma==0.:
		return gs
	gs = np.array(gs)
	conv_window = np.linspace(-dead_time_sigma*6,dead_time_sigma*6,int(math.ceil(dead_time_sigma*12/dt))+1)
	conv_window_size = conv_window.shape[0]
	conv_val = np.zeros_like(conv_window)
	conv_val[conv_window_size//2:] = normpdf(conv_window[conv_window_size//2:],0,dead_time_sigma)
	conv_val/=(np.sum(conv_val)*dt)
	cgs = []
	for g in gs:
		cgs.append(np.convolve(g,conv_val,mode=mode))
	cgs = np.array(cgs)
	a = int(0.5*(cgs.shape[1]-gs.shape[1]))
	if a==0:
		if cgs.shape[1]==gs.shape[1]:
			ret = cgs[:]
		else:
			ret = cgs[:,:-1]
	elif a==0.5*(cgs.shape[1]-gs.shape[1]):
		ret = cgs[:,a:-a]
	else:
		ret = cgs[:,a:-a-1]
	dead_time_shift = math.floor(dead_time/dt)
	output = np.zeros_like(ret)
	if dead_time_shift==0:
		output = ret
	else:
		output[:,dead_time_shift:] = ret[:,:-dead_time_shift]
	normalization = np.sum(output)*dt
	return tuple(output/normalization)

def fit(subject,method="full",time_units='seconds',T=None,dt=None,iti=None,tp=None,reward=1.,penalty=0.,n=101):
	dat,t,d = subject.load_data()
	mu,mu_indeces,count = np.unique((dat[:,0]-distractor)/ISI,return_inverse=True,return_counts=True)
	mus = np.concatenate((-mu[::-1],mu))
	counts = np.concatenate((count[::-1].astype(np.float64),count.astype(np.float64)))*0.5
	p = counts/np.sum(counts)
	
	prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	if time_units=='seconds':
		if T is None:
			T = 10.
		if iti is None:
			iti = 1.
		if tp is None:
			tp = 0.
		bounds = [np.array([0.,0.,0.,0.,0.]),np.array([10.,0.4,3.,1.,30.])]
		start_point = [0.2,0.1,0.5,0.1,0.2]
	else:
		if T is None:
			T = 10000.
		if iti is None:
			iti = 1000.
		if tp is None:
			tp = 0.
		bounds = [np.array([0.,0.,0.,0.,0.]),np.array([0.01,400,3000,1.,30.])]
		start_point = [0.0002,100,500,0.1,0.2]
	if dt is None:
		dt = ISI
	m = ct.DecisionPolicy(model_var=model_var,prior_mu_var=prior_mu_var,n=n,T=T,dt=dt,reward=reward,penalty=penalty,iti=iti,tp=tp,store_p=False)
	scaling_factor = bounds[1]-bounds[0]
	
	if method=="two_step":
		options = cma.CMAOptions({'bounds':[bounds[0][0],bounds[1][0]],'CMA_stds':scaling_factor[0]})
		res = (start_point[:4],None,None,None,None,None,None,)
		res2 = (start_point[:4],None,None,None,None,None,None,)
		#~ res = cma.fmin(two_step_merit, [start_point[0]], 1./3.,options,args=(m,dat,mu,mu_indeces),restarts=1)
		#~ res2 = []
		#~ two_step_merit(res[0][0],m,dat,mu,mu_indeces,res2)
		res = ({'cost':res2[0],'dead_time':res2[1],'dead_time_sigma':res2[2],'phase_out_prob':res2[3]},)+res[1:7]
	elif method=='full':
		options = cma.CMAOptions({'bounds':[bounds[0][:4],bounds[1][:4]],'CMA_stds':scaling_factor[:4]})
		res = (start_point[:4],None,None,None,None,None,None,)
		#~ res = cma.fmin(full_merit, start_point[:4], 1./3.,options,args=(m,dat,mu,mu_indeces),restarts=1)
		res = ({'cost':res[0][0],'dead_time':res[0][1],'dead_time_sigma':res[0][2],'phase_out_prob':res[0][3]},)+res[1:7]
	elif method=='confidence_only':
		f = open('fits/inference_fit_full_subject_'+str(subject.id)+'.pkl','r')
		out = pickle.load(f)
		f.close()
		options = cma.CMAOptions({'bounds':[bounds[0][-1],bounds[1][-1]],'CMA_stds':scaling_factor[-1]})
		res = ([start_point[-1]],None,None,None,None,None,None,)
		#~ res = cma.fmin(confidence_only_merit, [start_point[-1]], 1./3.,options,args=(m,dat,mu,mu_indeces,out[0]),restarts=1)
		res = ({'high_confidence_threshold':res[0][0]},)+res[1:7]
	elif method=='full_confidence':
		options = cma.CMAOptions({'bounds':bounds,'CMA_stds':scaling_factor})
		res = (start_point,None,None,None,None,None,None,)
		#~ res = cma.fmin(full_confidence_merit, start_point, 1./3.,options,args=(m,dat,mu,mu_indeces),restarts=1)
		res = ({'cost':res[0][0],'dead_time':res[0][1],'dead_time_sigma':res[0][2],'phase_out_prob':res[0][3],'high_confidence_threshold':res[0][4]},)+res[1:7]
	else:
		raise ValueError('Unknown method: {0}'.format(method))
	return res


def dead_time_merit(params,g1s,g2s,m,dat,mu,mu_indeces):
	dead_time = params[0]
	dead_time_sigma = params[1]
	phase_out_prob = params[2]
	if time_units=='seconds':
		max_RT = np.max(dat[:,1])*1e-3
	else:
		max_RT = np.max(dat[:,1])
	phase_out_likelihood = phase_out_prob/max_RT
	
	nlog_likelihood = 0.
	for index,(drift,g1,g2) in enumerate(zip(mu,g1s,g2s)):
		rt1_likelihood,rt2_likelihood = add_dead_time((g1,g2),m.dt,dead_time,dead_time_sigma)
		for drift_trial in dat[mu_indeces==index]:
			if time_units=='seconds':
				rt = drift_trial[1]*1e-3 # Response times are stored in ms while simulations assume times are written in seconds
			else:
				rt = drift_trial[1]
			perf = drift_trial[2]
			if perf==1:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(rt1_likelihood,rt))*(1-phase_out_prob)+0.5*phase_out_likelihood)
			else:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(rt2_likelihood,rt))*(1-phase_out_prob)+0.5*phase_out_likelihood)
	return nlog_likelihood

def two_step_merit(cost,m,dat,mu,mu_indeces,output_list=None):
	m.cost = cost[0]
	xub,xlb = m.xbounds()
	g1s = []
	g2s = []
	for drift in mu:
		g1,g2 = m.rt(drift,bounds=(xub,xlb))
		g1s.append(g1)
		g2s.append(g2)
	if time_units=='seconds':
		bounds = [np.array([0.,0.,0.,0.,0.]),np.array([10.,0.4,3.,1.,30.])]
		start_point = [0.2,0.1,0.5,0.1,0.2]
	else:
		bounds = [np.array([0.,0.,0.,0.,0.]),np.array([0.01,400,3000,1.,30.])]
		start_point = [0.0002,100,500,0.1,0.2]
	scaling_factor = bounds[1]-bounds[0]
	options = cma.CMAOptions({'bounds':[start_point[0][1:4],start_point[1][1:4]],'CMA_stds':scaling_factor[1:4]})
	res = cma.fmin(dead_time_merit, start_point[1:4], 1./3.,options,args=(g1s,g2s,m,dat,mu,mu_indeces),restarts=1)
	params = res[0]
	nlog_likelihood = res[1]
	#~ params,nlog_likelihood,niter,nfuncalls,warnflag =\
		#~ scipy.optimize.fmin(dead_time_merit,[1.,0.1],args=(g1s,g2s,m,dat,mu,mu_indeces),\
							#~ full_output=True,disp=False)
	if output_list is not None:
		del output_list[:]
		output_list.append(cost)
		output_list.extend(params)
	return nlog_likelihood

def full_merit(params,m,dat,mu,mu_indeces):
	cost = params[0]
	dead_time = params[1]
	dead_time_sigma = params[2]
	phase_out_prob = params[3]
	if time_units=='seconds':
		max_RT = np.max(dat[:,1])*1e-3
	else:
		max_RT = np.max(dat[:,1])
	
	nlog_likelihood = 0.
	m.cost = cost
	xub,xlb = m.xbounds()
	for index,drift in enumerate(mu):
		g1,g2 = add_dead_time(m.rt(drift,bounds=(xub,xlb)),m.dt,dead_time,dead_time_sigma)
		for drift_trial in dat[mu_indeces==index]:
			if time_units=='seconds':
				rt = drift_trial[1]*1e-3 # Response times are stored in ms while simulations assume times are written in seconds
			else:
				rt = drift_trial[1]
			perf = drift_trial[2]
			if perf==1:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(g1,rt))*(1-phase_out_prob)+0.5*phase_out_prob/max_RT)
			else:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(g2,rt))*(1-phase_out_prob)+0.5*phase_out_prob/max_RT)
	return nlog_likelihood

def confidence_only_merit(params,m,dat,mu,mu_indeces,decision_parameters):
	cost = decision_parameters['cost']
	dead_time = decision_parameters['dead_time']
	dead_time_sigma = decision_parameters['dead_time_sigma']
	phase_out_prob = decision_parameters['phase_out_prob']
	high_confidence_threshold = params[0]
	if time_units=='seconds':
		max_RT = np.max(dat[:,1])*1e-3
	else:
		max_RT = np.max(dat[:,1])
	
	nlog_likelihood = 0.
	m.cost = cost
	xub,xlb = m.xbounds()
	
	if time_units=='seconds' and m.dt>1e-3:
		_dt = 1e-3
	elif time_units=='milliseconds' and m.dt>1.:
		_dt = 1.
	else:
		_dt = None
	
	if _dt:
		_nT = int(m.T/_dt)+1
		_t = np.arange(0.,_nT,dtype=np.float64)*_dt
		log_odds = m.log_odds()
		log_odds = np.array([np.interp(_t,m.t,log_odds[0]),np.interp(_t,m.t,log_odds[0])])
	else:
		_nT = m.nT
		log_odds = m.log_odds()
	
	phigh = (1.-0.75*phase_out_prob)*np.ones((2,_nT))
	phigh[log_odds<high_confidence_threshold] = 0.25*phase_out_prob
	if _dt:
		ratio = np.ceil(_nT/m.nT)
		tail = _nT%m.nT
		if tail!=0:
			padded_phigh = np.concatenate((phigh,np.nan*np.ones((2,m.nT-tail),dtype=np.float)),axis=1)
		else:
			padded_phigh = phigh
		padded_phigh = np.reshape(padded_phigh,(2,-1,ratio))
		phigh = np.nanmean(padded_phigh,axis=2)
	plow = 1.-phigh
	
	for index,drift in enumerate(mu):
		gs = np.array(m.rt(drift,bounds=(xub,xlb)))
		confidence_rt = np.concatenate((phigh*gs,plow*gs))
		g1h,g2h,g1l,g2l = add_dead_time(confidence_rt,m.dt,dead_time,dead_time_sigma)
		gh = g1h+g2h
		gl = g1l+g2l
		for drift_trial in dat[mu_indeces==index]:
			if time_units=='seconds':
				rt = drift_trial[1]*1e-3 # Response times are stored in ms while simulations assume times are written in seconds
			else:
				rt = drift_trial[1]
			conf = drift_trial[3]
			if conf==2:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(gh,rt))*(1-phase_out_prob)+0.5*phase_out_prob/max_RT)
			else:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(gl,rt))*(1-phase_out_prob)+0.5*phase_out_prob/max_RT)
	return nlog_likelihood


def full_confidence_merit(params,m,dat,mu,mu_indeces):
	cost = params[0]
	dead_time = params[1]
	dead_time_sigma = params[2]
	phase_out_prob = params[3]
	high_confidence_threshold = params[4]
	if time_units=='seconds':
		max_RT = np.max(dat[:,1])*1e-3
	else:
		max_RT = np.max(dat[:,1])
	
	nlog_likelihood = 0.
	m.cost = cost
	xub,xlb = m.xbounds()
	
	if time_units=='seconds' and m.dt>1e-3:
		_dt = 1e-3
	elif time_units=='milliseconds' and m.dt>1.:
		_dt = 1.
	else:
		_dt = None
	
	if _dt:
		_nT = int(m.T/_dt)+1
		_t = np.arange(0.,_nT,dtype=np.float64)*_dt
		log_odds = m.log_odds()
		log_odds = np.array([np.interp(_t,m.t,log_odds[0]),np.interp(_t,m.t,log_odds[0])])
	else:
		_nT = m.nT
		log_odds = m.log_odds()
	
	phigh = (1.-0.75*phase_out_prob)*np.ones((2,_nT))
	phigh[log_odds<high_confidence_threshold] = 0.25*phase_out_prob
	if _dt:
		ratio = np.ceil(_nT/m.nT)
		tail = _nT%m.nT
		if tail!=0:
			padded_phigh = np.concatenate((phigh,np.nan*np.ones((2,m.nT-tail),dtype=np.float)),axis=1)
		else:
			padded_phigh = phigh
		padded_phigh = np.reshape(padded_phigh,(2,-1,ratio))
		phigh = np.nanmean(padded_phigh,axis=2)
	plow = 1.-phigh
	
	for index,drift in enumerate(mu):
		gs = np.array(m.rt(drift,bounds=(xub,xlb)))
		confidence_rt = np.concatenate((phigh*gs,plow*gs))
		g1h,g2h,g1l,g2l = add_dead_time(confidence_rt,m.dt,dead_time,dead_time_sigma)
		for drift_trial in dat[mu_indeces==index]:
			if time_units=='seconds':
				rt = drift_trial[1]*1e-3 # Response times are stored in ms while simulations assume times are written in seconds
			else:
				rt = drift_trial[1]
			perf = drift_trial[2]
			conf = drift_trial[3]
			if perf==1 and conf==2:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(g1h,rt))*(1-phase_out_prob)+0.25*phase_out_prob/max_RT)
			elif perf==0 and conf==2:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(g2h,rt))*(1-phase_out_prob)+0.25*phase_out_prob/max_RT)
			elif perf==1 and conf==1:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(g1l,rt))*(1-phase_out_prob)+0.25*phase_out_prob/max_RT)
			else:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(g2l,rt))*(1-phase_out_prob)+0.25*phase_out_prob/max_RT)
	return nlog_likelihood

def theoretical_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,confidence_params=None,include_t0=True):
	rt,dec_gs = decision_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,return_gs=True,include_t0=include_t0)
	if confidence_params:
		dec_conf = confidence_rt_distribution(dec_gs,cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,confidence_params,include_t0=include_t0)
		for key in rt.keys():
			rt[key].update(dec_conf[key])
	return rt

def decision_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,return_gs=False,include_t0=True):
	if include_t0:
		rt = {'full':{'all':np.zeros((2,m.t.shape[0]))}}
		gs = {'full':{'all':np.zeros((2,m.t.shape[0]))}}
		phased_out_rt = np.zeros_like(m.t)
	else:
		rt = {'full':{'all':np.zeros((2,m.t.shape[0]-1))}}
		gs = {'full':{'all':np.zeros((2,m.t.shape[0]-1))}}
		phased_out_rt = np.zeros_like(m.t)[1:]
	m.cost = cost
	_max_RT = m.t[np.ceil(max_RT/m.dt)]
	_dead_time = m.t[np.floor(dead_time/m.dt)]
	if include_t0:
		phased_out_rt[m.t<_max_RT] = 1./(_max_RT)
		#~ phased_out_rt[np.logical_and(m.t<_max_RT,m.t>_dead_time)] = 1./(_max_RT-_dead_time)
	else:
		phased_out_rt[m.t[1:]<_max_RT] = 1./(_max_RT)
		#~ phased_out_rt[np.logical_and(m.t[1:]<_max_RT,m.t[1:]>_dead_time)] = 1./(_max_RT-_dead_time)
	xub,xlb = m.xbounds()
	for index,drift in enumerate(mu):
		g = np.array(m.rt(drift,bounds=(xub,xlb)))
		if not include_t0:
			g = g[:,1:]
		g1,g2 = add_dead_time(g,m.dt,dead_time,dead_time_sigma)
		g1 = g1*(1-phase_out_prob)+0.5*phase_out_prob*phased_out_rt
		g2 = g2*(1-phase_out_prob)+0.5*phase_out_prob*phased_out_rt
		rt[drift] = {}
		rt[drift]['all'] = np.array([g1,g2])
		rt['full']['all']+=rt[drift]['all']*mu_prob[index]
		gs[drift] = {}
		gs[drift]['all'] = np.array(g)
		gs['full']['all']+=gs[drift]['all']*mu_prob[index]
	output = (rt,)
	if return_gs:
		output+=(gs,)
	return output

def confidence_rt_distribution(dec_gs,cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,confidence_params,include_t0=True):
	if include_t0:
		rt = {'full':{'high':np.zeros((2,m.t.shape[0])),'low':np.zeros((2,m.t.shape[0]))}}
		phased_out_rt = np.zeros_like(m.t)
	else:
		rt = {'full':{'high':np.zeros((2,m.t.shape[0]-1)),'low':np.zeros((2,m.t.shape[0]-1))}}
		phased_out_rt = np.zeros_like(m.t)[1:]
	m.cost = cost
	_max_RT = m.t[np.ceil(max_RT/m.dt)]
	_dead_time = m.t[np.floor(dead_time/m.dt)]
	
	if time_units=='seconds' and m.dt>1e-3:
		_dt = 1e-3
	elif time_units=='milliseconds' and m.dt>1.:
		_dt = 1.
	else:
		_dt = None
	
	orig_phigh = (1.-0.75*phase_out_prob)*np.ones((2,m.nT))
	orig_phigh[m.log_odds()<confidence_params[0]] = 0.25*phase_out_prob
	if _dt:
		_nT = int(m.T/_dt)+1
		_t = np.arange(0.,_nT,dtype=np.float64)*_dt
		log_odds = m.log_odds()
		log_odds = np.array([np.interp(_t,m.t,log_odds[0]),np.interp(_t,m.t,log_odds[0])])
	else:
		_nT = m.nT
		log_odds = m.log_odds()
	
	phased_out_rt[m.t<_max_RT] = 1./(_max_RT)
	#~ phased_out_rt[np.logical_and(m.t<_max_RT,m.t>_dead_time)] = 1./(_max_RT-_dead_time)
	
	phigh = (1.-0.75*phase_out_prob)*np.ones((2,_nT))
	phigh[log_odds<confidence_params[0]] = 0.25*phase_out_prob
	if _dt:
		ratio = int(np.ceil(_nT/m.nT))
		tail = _nT%m.nT
		if tail!=0:
			padded_phigh = np.concatenate((phigh,np.nan*np.ones((2,m.nT-tail),dtype=np.float)),axis=1)
		else:
			padded_phigh = phigh
		padded_phigh = np.reshape(padded_phigh,(2,-1,ratio))
		phigh = np.nanmean(padded_phigh,axis=2)
	plow = 1.-phigh
	
	for index,drift in enumerate(mu):
		g = dec_gs[drift]['all']
		if include_t0:
			gh=phigh*g
			gl=plow*g
		else:
			gh=phigh[:,1:]*g
			gl=plow[:,1:]*g
		g1h,g2h,g1l,g2l = add_dead_time(np.concatenate((gh,gl)),m.dt,dead_time,dead_time_sigma)
		g1h = g1h*(1-0.75*phase_out_prob)+0.25*phase_out_prob*phased_out_rt
		g2h = g2h*(1-0.75*phase_out_prob)+0.25*phase_out_prob*phased_out_rt
		g1l = g1l*(1-0.75*phase_out_prob)+0.25*phase_out_prob*phased_out_rt
		g2l = g2l*(1-0.75*phase_out_prob)+0.25*phase_out_prob*phased_out_rt
		rt[drift] = {}
		rt[drift]['high'] = np.array([g1h,g2h])
		rt[drift]['low'] = np.array([g1l,g2l])
		rt['full']['high']+= rt[drift]['high']*mu_prob[index]
		rt['full']['low']+= rt[drift]['low']*mu_prob[index]
	return rt

def plot_fit(subject,method='full',save=None):
	f = open('fits/inference_fit_'+method+'_subject_'+str(subject.id)+'.pkl','r')
	out = pickle.load(f)
	f.close()
	if isinstance(out,dict):
		fit_output = out['fit_output']
		options = out['options']
		cost = fit_output[0]['cost']
		dead_time = fit_output[0]['dead_time']
		dead_time_sigma = fit_output[0]['dead_time_sigma']
		phase_out_prob = fit_output[0]['phase_out_prob']
		try:
			high_conf_thresh = fit_output[0]['high_confidence_threshold']
		except KeyError:
			try:
				f = open("fits/inference_fit_confidence_only_subject_"+str(subject.id)+".pkl",'r')
				print f
				out2 = pickle.load(f)
				f.close()
				print out2
				if isinstance(out2,dict):
					out2 = out2['fit_output']
				high_conf_thresh = out2[0]['high_confidence_threshold']
			except (IOError,EOFError):
				hand_picked_thresh = [0.77,0.6,0.79,0.61,0.62,0.90,0.81]
				high_conf_thresh = hand_picked_thresh[subject.id]
			except Exception as err:
				high_conf_thresh = out2[0][0]
	else:
		options = {'time_units':'seconds','T':10.,'iti':1.,'tp':0.,'dt':ISI,'reward':1,'penalty':0,'n':101}
		try:
			cost = out[0]['cost']
			dead_time = out[0]['dead_time']
			dead_time_sigma = out[0]['dead_time_sigma']
			phase_out_prob = out[0]['phase_out_prob']
			try:
				high_conf_thresh = out[0]['high_confidence_threshold']
			except KeyError:
				try:
					f = open("fits/inference_fit_confidence_only_subject_"+str(subject.id)+".pkl",'r')
					out2 = pickle.load(f)
					f.close()
					high_conf_thresh = out2[0]['high_confidence_threshold']
				except (IOError,EOFError):
					hand_picked_thresh = [0.77,0.6,0.79,0.61,0.62,0.90,0.81]
					high_conf_thresh = hand_picked_thresh[subject.id]
				except Exception as err:
					high_conf_thresh = out2[0][0]
		except IndexError,TypeError:
			cost = out[0][0]
			dead_time = out[0][1]
			dead_time_sigma = out[0][2]
			phase_out_prob = out[0][3]
			if 'confidence' in method:
				high_conf_thresh = out[0][4]
	time_units = options['time_units']
	
	dat,t,d = subject.load_data()
	if time_units=='seconds':
		rt = dat[:,1]*1e-3
	else:
		rt = dat[:,1]
	max_RT = np.max(rt)
	perf = dat[:,2]
	conf = dat[:,3]
	temp,edges = np.histogram(rt,100)
	
	high_hit_rt,temp = np.histogram(rt[np.logical_and(perf==1,conf==2)],edges)
	low_hit_rt,temp = np.histogram(rt[np.logical_and(perf==1,conf==1)],edges)
	high_miss_rt,temp = np.histogram(rt[np.logical_and(perf==0,conf==2)],edges)
	low_miss_rt,temp = np.histogram(rt[np.logical_and(perf==0,conf==1)],edges)
	
	high_hit_rt = high_hit_rt.astype(np.float64)
	low_hit_rt = low_hit_rt.astype(np.float64)
	high_miss_rt = high_miss_rt.astype(np.float64)
	low_miss_rt = low_miss_rt.astype(np.float64)
	
	hit_rt = high_hit_rt+low_hit_rt
	miss_rt = high_miss_rt+low_miss_rt
	
	xh = np.array([0.5*(x+y) for x,y in zip(edges[1:],edges[:-1])])
	
	normalization = np.sum(hit_rt+miss_rt)*(xh[1]-xh[0])
	hit_rt/=normalization
	miss_rt/=normalization
	
	high_hit_rt/=normalization
	high_miss_rt/=normalization
	low_hit_rt/=normalization
	low_miss_rt/=normalization
	
	mu,mu_indeces,count = np.unique((dat[:,0]-distractor)/ISI,return_inverse=True,return_counts=True)
	mu_prob = count.astype(np.float64)
	mu_prob/=np.sum(mu_prob)
	mus = np.concatenate((-mu[::-1],mu))
	counts = np.concatenate((count[::-1].astype(np.float64),count.astype(np.float64)))*0.5
	p = counts/np.sum(counts)
	prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	
	T = options['T']
	dt = options['dt']
	iti = options['iti']
	tp = options['tp']
	reward = options['reward']
	penalty = options['penalty']
	n = options['n']
	if time_units=='seconds':
		if T is None:
			T = 10.
		if iti is None:
			iti = 1.
		if tp is None:
			tp = 0.
	else:
		if T is None:
			T = 10000.
		if iti is None:
			iti = 1000.
		if tp is None:
			tp = 0.
	if dt is None:
		dt = ISI
	m = ct.DecisionPolicy(model_var=model_var,prior_mu_var=prior_mu_var,n=n,T=T,dt=dt,reward=reward,penalty=penalty,iti=iti,tp=tp,store_p=False)
	
	
	#~ print subject.id,[cost,dead_time,dead_time_sigma,phase_out_prob,high_conf_thresh]
	#~ print prior_mu_var,model_var,ISI
	#~ print full_confidence_merit([cost,dead_time,dead_time_sigma,phase_out_prob,high_conf_thresh],m,dat,mu,mu_indeces)
	#~ high_conf_thresh = 0.3
	#~ print full_confidence_merit([cost,dead_time,dead_time_sigma,phase_out_prob,high_conf_thresh],m,dat,mu,mu_indeces)
	confidence_params = [high_conf_thresh]
	sim_rt = theoretical_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,confidence_params)
	
	mxlim = np.ceil(max_RT)
	mt.rc('axes', color_cycle=['b','r'])
	plt.figure(figsize=(11,8))
	plt.subplot(121)
	plt.step(xh,hit_rt,label='Subject '+str(subject.id)+' hit rt',where='post')
	plt.step(xh,-miss_rt,label='Subject '+str(subject.id)+' miss rt',where='post')
	plt.plot(m.t,sim_rt['full']['all'][0],label='Theoretical hit rt',linewidth=2)
	plt.plot(m.t,-sim_rt['full']['all'][1],label='Theoretical miss rt',linewidth=2)
	plt.xlim([0,mxlim])
	if time_units=='seconds':
		plt.xlabel('T [s]')
	else:
		plt.xlabel('T [ms]')
	plt.ylabel('Prob density')
	plt.legend()
	plt.subplot(122)
	plt.step(xh,high_hit_rt+high_miss_rt,label='Subject '+str(subject.id)+' high',where='post')
	plt.step(xh,-(low_hit_rt+low_miss_rt),label='Subject '+str(subject.id)+' low',where='post')
	plt.plot(m.t,np.sum(sim_rt['full']['high'],axis=0),label='Theoretical high',linewidth=2)
	plt.plot(m.t,-np.sum(sim_rt['full']['low'],axis=0),label='Theoretical low',linewidth=2)
	plt.xlim([0,mxlim])
	if time_units=='seconds':
		plt.xlabel('T [s]')
	else:
		plt.xlabel('T [ms]')
	plt.legend()
	
	
	#~ plt.subplot(223)
	#~ plt.plot(m.t,m.log_odds().T)
	#~ plt.plot([m.t[0],m.t[-1]],[high_conf_thresh,high_conf_thresh],'--k')
	#~ 
	#~ plt.subplot(222)
	#~ plt.plot(xh,high_hit_rt,label='Subject '+str(subject.id)+' high hit')
	#~ plt.plot(xh,-high_miss_rt,label='Subject '+str(subject.id)+' high miss')
	#~ plt.plot(m.t,sim_rt['full']['high'][0],label='Theoretical high hit')
	#~ plt.plot(m.t,-sim_rt['full']['high'][1],label='Theoretical high miss')
	#~ plt.xlim([0,mxlim])
	#~ plt.legend()
	#~ plt.subplot(224)
	#~ plt.plot(xh,low_hit_rt,label='Subject '+str(subject.id)+' high hit')
	#~ plt.plot(xh,-low_miss_rt,label='Subject '+str(subject.id)+' high miss')
	#~ plt.plot(m.t,sim_rt['full']['low'][0],label='Theoretical high hit')
	#~ plt.plot(m.t,-sim_rt['full']['low'][1],label='Theoretical high miss')
	#~ plt.xlim([0,mxlim])
	#~ plt.xlabel('T [s]')
	#~ plt.legend()
	if save:
		save.savefig()
	else:
		plt.show(True)

def parse_input():
	script_help = """ moving_bounds_fits.py help
 Sintax:
 moving_bounds_fits.py [option flag] [option value]
 
 moving_bounds_fits.py -h [or --help] displays help
 
 Optional arguments are:
 '-t' or '--task': Integer that identifies the task number when running multiple tasks in parallel. Is one based, thus the first task is task 1 [default 1]
 '-nt' or '--ntasks': Integer that identifies the number tasks working in parallel [default 1]
 '-m' or '--method': String that identifies the fit method. Available values are two_step, full, confidence_only and full_confidence. [default full]
 '-s' or '--save': This flag takes no values. If present it saves the figure
 '-u' or '--units': String that identifies the time units that will be used. Available values are seconds and milliseconds. [default seconds]
 '-n': Integer that specifies the belief space discretization for the DecisionPolicy instance. Must be an uneven number, if an even number is supplied it will be recast to closest, larger uneven integer (e.g. if n=100 then it will be casted to n=101) [Default 101]
 '-T': Float that specifies the maximum time for the DecisionPolicy instance. [Default 10 seconds]
 '-dt': Float that specifies the time step for the DecisionPolicy instance. [Default 0.04 seconds]
 '-iti': Float that specifies the inter trial time for the DecisionPolicy instance. [Default 1 seconds]
 '-tp': Float that specifies the penalty time for the DecisionPolicy instance. [Default 0 seconds]
 '-r' or '--reward': Float that specifies the reward for the DecisionPolicy instance. [Default 10 seconds]
 '-p' or '--penalty': Float that specifies the penalty for the DecisionPolicy instance. [Default 10 seconds]
 
 Example:
 python moving_bounds_fits.py -T 10 -dt 0.001 --save"""
	options =  {'task':1,'ntasks':1,'method':'full','save':None,'time_units':'seconds',
				'T':None,'iti':None,'tp':None,'dt':None,'reward':1,'penalty':0,'n':101}
	
	expecting_key = True
	key = None
	for i,arg in enumerate(sys.argv[1:]):
		if expecting_key:
			if arg=='-t' or arg=='--task':
				key = 'task'
				expecting_key = False
			elif arg=='-nt' or arg=='--ntasks':
				key = 'ntasks'
				expecting_key = False
			elif arg=='-m' or arg=='--method':
				key = 'method'
				expecting_key = False
			elif arg=='-s' or arg=='--save':
				options['save'] = True
			elif arg=='-u' or arg=='--units':
				key = 'time_units'
				expecting_key = False
			elif arg=='-n':
				key = 'n'
				expecting_key = False
			elif arg=='-T':
				key = 'T'
				expecting_key = False
			elif arg=='-dt':
				key = 'dt'
				expecting_key = False
			elif arg=='-iti':
				key = 'iti'
				expecting_key = False
			elif arg=='-tp':
				key = 'tp'
				expecting_key = False
			elif arg=='-r' or arg=='--reward':
				key = 'reward'
				expecting_key = False
			elif arg=='-p' or arg=='--penalty':
				key = 'penalty'
				expecting_key = False
			elif arg=='-h' or arg=='--help':
				print script_help
				sys.exit()
			else:
				raise RuntimeError("Unknown option: {opt} encountered in position {pos}. Refer to the help to see the list of options".format({'opt':arg,'pos':i+1}))
		else:
			expecting_key = True
			if key in ['n','task','ntasks']:
				options[key] = int(arg)
			elif key in ['T','dt','iti','tp','reward','penalty']:
				options[key] = float(arg)
			else:
				options[key] = arg
	# Shift task from 1 base to 0 based
	options['task']-=1
	if options['time_units'] not in ['seconds','milliseconds']:
		raise ValueError("Unknown supplied units: '{units}'. Available values are seconds and milliseconds".format({'units':options['time_units']}))
	if options['method'] not in ['two_step','full','confidence_only','full_confidence']:
		raise ValueError("Unknown supplied method: '{method}'. Available values are two_step, full, confidence_only and full_confidence".format({'method':options['method']}))
	return options

if __name__=="__main__":
	options = parse_input()
	method = options['method']
	save = options['save']
	task = options['task']
	ntasks = options['ntasks']
	
	set_time_units(options['time_units'])
	subjects = io.unique_subjects(data_dir)
	subjects.append(io.merge_subjects(subjects))
	for i,s in enumerate(subjects):
		if (i-task)%ntasks==0:
			#~ fit_output = fit(s,method=method,time_units=options['time_units'],n=options['n'],T=options['T'],dt=options['dt'],iti=options['iti'],tp=options['tp'],reward=options['reward'],penalty=options['penalty'])
			#~ f = open("fits/inference_fit_"+method+"_subject_"+str(s.id)+".pkl",'w')
			#~ pickle.dump({'fit_output':fit_output,'options':options},f,pickle.HIGHEST_PROTOCOL)
			#~ f.close()
			#~ if method=='full' or method=='two_step':
				#~ fit_output = fit(s,method='confidence_only',time_units=options['time_units'],n=options['n'],T=options['T'],dt=options['dt'],iti=options['iti'],tp=options['tp'],reward=options['reward'],penalty=options['penalty'])
				#~ f = open("fits/inference_fit_confidence_only_subject_"+str(s.id)+".pkl",'w')
				#~ pickle.dump({'fit_output':fit_output,'options':options},f,pickle.HIGHEST_PROTOCOL)
				#~ f.close()
			plot_fit(s,method=method,save=save)
			#~ break
	if save:
		save.close()
