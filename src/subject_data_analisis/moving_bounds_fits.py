from __future__ import division

import enum, os, sys, math, scipy, pickle, cma
import data_io as io
import cost_time as ct
import numpy as np
from utils import normpdf
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
ISI = 0.04 #seconds
distractor = 50. # cd/m^2
patch_sigma = 5. # cd/m^2
model_var = (patch_sigma**2)*2/ISI

def add_dead_time(gs,dt,dead_time,dead_time_sigma,mode='full'):
	"""
	new_gs = add_dead_time(gs,dt,dead_time_sigma,mode='full')
	"""
	if dead_time_sigma==0.:
		return gs
	g1,g2 = gs
	conv_window = np.linspace(-dead_time_sigma*6,dead_time_sigma*6,int(math.ceil(dead_time_sigma*12/dt))+1)
	conv_window_size = conv_window.shape[0]
	conv_val = np.zeros_like(conv_window)
	conv_val[conv_window_size//2:] = normpdf(conv_window[conv_window_size//2:],0,dead_time_sigma)
	conv_val/=(np.sum(conv_val)*dt)
	#~ print "g1 = ",g1
	#~ print "conv_val = ", conv_val
	cg1 = np.convolve(g1,conv_val,mode=mode)
	cg2 = np.convolve(g2,conv_val,mode=mode)
	a = int(0.5*(cg1.shape[0]-g1.shape[0]))
	if a==0:
		if cg1.shape[0]==g1.shape[0]:
			ret = np.array([cg1,cg2])
		else:
			ret = np.array([cg1[:-1],cg2[:-1]])
	elif a==0.5*(cg1.shape[0]-g1.shape[0]):
		ret = np.array([cg1[a:-a],cg2[a:-a]])
	else:
		ret = np.array([cg1[a:-a-1],cg2[a:-a-1]])
	dead_time_shift = math.floor(dead_time/dt)
	output = np.zeros_like(ret)
	if dead_time_shift==0:
		output = ret
	else:
		output[:,dead_time_shift:] = ret[:,:-dead_time_shift]
	normalization = np.sum(output)*dt
	return tuple(output/normalization)

def log_odds_likelihood(gs,confidence_params,log_odds,dt,dead_time,dead_time_sigma,phase_out_prob):
	if len(confidence_params)==1:
		p = (1.-phase_out_prob)*np.ones_like(gs)
		p[log_odds<confidence_params[0]] = 0.
		return add_dead_time(p,dt,dead_time,dead_time_sigma)

def fit(subject,method="full"):
	dat,t,d = subject.load_data()
	mu,mu_indeces,count = np.unique((dat[:,0]-distractor)/ISI,return_inverse=True,return_counts=True)
	mus = np.concatenate((-mu[::-1],mu))
	counts = np.concatenate((count[::-1].astype(np.float64),count.astype(np.float64)))*0.5
	p = counts/np.sum(counts)
	
	prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	
	m = ct.DecisionPolicy(model_var=model_var,prior_mu_var=prior_mu_var,n=101,T=10,dt=ISI,reward=1,penalty=0,iti=1.,tp=0.,store_p=False)
	if method=="two_step":
		options = cma.CMAOptions({'bounds':[0.,10.]})
		res = cma.fmin(two_step_merit, [0.2], 1./3.,options,args=(m,dat,mu,mu_indeces),restarts=1)
		res2 = []
		two_step_merit(res[0][0],m,dat,mu,mu_indeces,res2)
		res[0] = {'cost':res2[0],'dead_time':res2[1],'dead_time_sigma':res2[2],'phase_out_prob':res2[3]}
		return res[:7]
	elif method=='full':
		options = cma.CMAOptions({'bounds':[np.array([0.,0.,0.,0.]),np.array([10.,0.4,3.,1.])]})
		res = cma.fmin(full_merit, [0.2,0.1,0.5,0.1], 1./3.,options,args=(m,dat,mu,mu_indeces),restarts=1)
		res[0] = {'cost':res[0][0],'dead_time':res[0][1],'dead_time_sigma':res[0][2],'phase_out_prob':res[0][3]}
		return res[:7]
	else:
		raise ValueError('Unknown method: {0}'.format(method))

def dead_time_merit(params,g1s,g2s,m,dat,mu,mu_indeces):
	dead_time = params[0]
	dead_time_sigma = params[1]
	phase_out_prob = params[2]
	max_RT = np.max(dat[:,1])*1e-3
	phase_out_likelihood = phase_out_prob/max_RT
	
	nlog_likelihood = 0.
	for index,(drift,g1,g2) in enumerate(zip(mu,g1s,g2s)):
		rt1_likelihood,rt2_likelihood = add_dead_time((g1,g2),m.dt,dead_time,dead_time_sigma)
		for drift_trial in dat[mu_indeces==index]:
			rt = drift_trial[1]*1e-3 # Response times are stored in ms while simulations assume times are written in seconds
			perf = drift_trial[2]
			if perf==1:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(rt1_likelihood,rt))*(1-phase_out_prob)+phase_out_likelihood)
			else:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(rt2_likelihood,rt))*(1-phase_out_prob)+phase_out_likelihood)
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
	options = cma.CMAOptions({'bounds':[np.array([0.,0.,0.]),np.array([0.5,3.,1.])]})
	res = cma.fmin(dead_time_merit, [0.1,0.5,0.1], 1./3.,options,args=(g1s,g2s,m,dat,mu,mu_indeces),restarts=1)
	params = res[0]
	nlog_likelihood = res[1]
	#~ params,nlog_likelihood,niter,nfuncalls,warnflag =\
		#~ scipy.optimize.fmin(dead_time_merit,[1.,0.1],args=(g1s,g2s,m,dat,mu,mu_indeces),\
							#~ full_output=True,disp=False)
	if output_list is not None:
		del output_list[:]
		output_list.append(cost)
		output_list.extend(params)
	print cost[0],params[0],params[1],nlog_likelihood
	return nlog_likelihood

def full_merit(params,m,dat,mu,mu_indeces):
	cost = params[0]
	dead_time = params[1]
	dead_time_sigma = params[2]
	phase_out_prob = params[3]
	max_RT = np.max(dat[:,1])*1e-3
	
	nlog_likelihood = 0.
	m.cost = cost
	xub,xlb = m.xbounds()
	for index,drift in enumerate(mu):
		g1,g2 = add_dead_time(m.rt(drift,bounds=(xub,xlb)),m.dt,dead_time,dead_time_sigma)
		for drift_trial in dat[mu_indeces==index]:
			rt = drift_trial[1]*1e-3 # Response times are stored in ms while simulations assume times are written in seconds
			perf = drift_trial[2]
			if perf==1:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(g1,rt))*(1-phase_out_prob)+phase_out_prob/max_RT)
			else:
				nlog_likelihood-=np.log(np.exp(-m.rt_nlog_like(g2,rt))*(1-phase_out_prob)+phase_out_prob/max_RT)
	print nlog_likelihood
	return nlog_likelihood

def theoretical_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT):
	rt = {'full':np.zeros((2,m.t.shape[0]))}
	m.cost = cost
	phased_out_rt = np.zeros_like(m.t)
	_max_RT = m.t[np.ceil(max_RT/m.dt)]
	_dead_time = m.t[np.floor(dead_time/m.dt)]
	phased_out_rt[m.t<_max_RT] = 1./(_max_RT)
	#~ phased_out_rt[np.logical_and(m.t<_max_RT,m.t>_dead_time)] = 1./(_max_RT-_dead_time)
	xub,xlb = m.xbounds()
	for index,drift in enumerate(mu):
		g1,g2 = add_dead_time(m.rt(drift,bounds=(xub,xlb)),m.dt,dead_time,dead_time_sigma)
		g1 = g1*(1-phase_out_prob)+0.5*phase_out_prob*phased_out_rt
		g2 = g2*(1-phase_out_prob)+0.5*phase_out_prob*phased_out_rt
		rt[drift] = np.array([g1,g2])
		rt['full']+=rt[drift]*mu_prob[index]
	return rt

def plot_fit(subject,method='full',save=None):
	f = open('fits/inference_fit_'+method+'_subject_'+str(subject.id)+'.pkl','r')
	out = pickle.load(f)
	try:
		cost = out[0]['cost']
		dead_time = out[0]['dead_time']
		dead_time_sigma = out[0]['dead_time_sigma']
		phase_out_prob = out[0]['phase_out_prob']
	except TypeError:
		cost = out[0][0]
		dead_time = 0.
		dead_time_sigma = out[0][1]
		phase_out_prob = out[0][2]
	
	dat,t,d = subject.load_data()
	rt = dat[:,1]*1e-3
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
	
	high_hit_rt/=(np.sum(high_hit_rt+low_hit_rt)*(xh[1]-xh[0]))
	high_miss_rt/=(np.sum(high_miss_rt+low_miss_rt)*(xh[1]-xh[0]))
	
	mu,mu_indeces,count = np.unique((dat[:,0]-distractor)/ISI,return_inverse=True,return_counts=True)
	mu_prob = count.astype(np.float64)
	mu_prob/=np.sum(mu_prob)
	mus = np.concatenate((-mu[::-1],mu))
	counts = np.concatenate((count[::-1].astype(np.float64),count.astype(np.float64)))*0.5
	p = counts/np.sum(counts)
	prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	m = ct.DecisionPolicy(model_var=model_var,prior_mu_var=prior_mu_var,n=101,T=10,dt=ISI,reward=1,penalty=0,iti=1.,tp=0.,store_p=False)
	
	sim_rt = theoretical_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT)
	
	plt.figure()
	plt.subplot(121)
	plt.plot(xh,hit_rt,label='Subject '+str(subject.id)+' hit rt')
	plt.plot(xh,-miss_rt,label='Subject '+str(subject.id)+' miss rt')
	plt.plot(m.t,sim_rt['full'][0],label='Theorical hit rt')
	plt.plot(m.t,-sim_rt['full'][1],label='Theorical miss rt')
	plt.xlim([0,5])
	plt.xlabel('T [s]')
	plt.ylabel('Prob density')
	plt.legend()
	
	log_odds = m.log_odds()
	high_conf_thresh = 0.6
	confidence_params = [high_conf_thresh]
	sim_high_likelihood = log_odds_likelihood(sim_rt['full'],confidence_params,log_odds,m.dt,dead_time,dead_time_sigma,phase_out_prob)
	print np.sum(sim_high_likelihood)*m.dt
	plt.subplot(122)
	plt.plot(xh,high_hit_rt,label='Subject '+str(subject.id)+' high hit')
	plt.plot(xh,-high_miss_rt,label='Subject '+str(subject.id)+' high miss')
	plt.plot(m.t,sim_high_likelihood[0],label='Theorical high hit')
	plt.plot(m.t,-sim_high_likelihood[1],label='Theorical high miss')
	if save:
		save.savefig()
	else:
		plt.show(True)

if __name__=="__main__":
	task = 0
	ntasks = 1
	method = 'full'
	save = None
	if len(sys.argv)>1:
		task = int(sys.argv[1])
		ntasks = int(sys.argv[2])
		if loc==Location.cluster:
			task-=1
		if len(sys.argv)>3:
			method = sys.argv[3]
			if len(sys.argv)>4:
				save = PdfPages('../../figs/inference_fits_'+method+'.pdf')
	
	subjects = io.unique_subjects(data_dir)
	subjects.append(io.merge_subjects(subjects))
	for i,s in enumerate(subjects):
		if (i-task)%ntasks==0:
			#~ f = open("inference_fit_"+method+"_subject_"+str(s.id)+".pkl",'w')
			#~ pickle.dump(fit(s,method=method),f,pickle.HIGHEST_PROTOCOL)
			#~ f.close()
			plot_fit(s,method=method,save=save)
			#~ break
	if save:
		save.close()
