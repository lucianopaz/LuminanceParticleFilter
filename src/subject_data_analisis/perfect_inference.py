#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package for fitting perfect inference behavior """

from __future__ import division
import numpy as np
import data_io as io
import kernels as ke
import os, itertools, sys, random, math, cma

class onlineMeanVar():
	def __init__(self):
		self.reset()
	def reset(self):
		self.n = 0
		self.delta = 0.
		self.mean = 0.
		self.M2 = 0.
		self.var = 0.
	def update(self,v):
		self.n+=1
		self.delta = v-self.mean
		self.mean+= self.delta/self.n
		self.M2+= self.delta*(v-self.mean)
		self.var = self.M2/(self.n-1) if n>1 else (0. if not isinstance(self.mean,np.ndarray) else np.zeros_like(self.mean))
	def val(self):
		return (self.mean,self.var)
	def add_value(self,value):
		self.update(value)
		return self.val()
	def arr_add_value(self,val_arr):
		mean = np.zeros_like(val_arr)
		var =  np.zeros_like(val_arr)
		for i,val in enumerate(val_arr):
			mean[i],var[i] = self.add_value(val)
		return mean,var

class PerfectInference(object):
	"""
	Virtual base class for perfect inference methods. Subclasses
	implement perfect inference with different known distribution
	variables
	"""
	def onlineInference(self,targetSignal,distractorSignal):
		pass
	def batchInference(self,targetSignalArr,distractorSignalArr):
		pass

class KnownVarPerfectInference(PerfectInference):
	"""
	Class that implements perfect bayes inference on the mean luminance
	classification task with known model variance
	"""
	def __init__(self,model_sigma_t=1.,model_sigma_d=1.,prior_mu_t=0.,prior_mu_d=0.,prior_va_t=1.,prior_va_d=1.,\
				threshold=5.):
		self.sigma_t = model_sigma_t
		self.sigma_d = model_sigma_d
		self.prior_mu_t = prior_mu_t
		self.prior_mu_d = prior_mu_d
		self.prior_va_t = prior_va_t
		self.prior_va_d = prior_va_d
		
		self.threshold = threshold
		self.signals = []
	
	def onlineInference(self,targetSignal,distractorSignal,storeSignals=False,timeout=np.inf):
		n = 0
		ct = 0.
		cd = 0.
		decided = False
		it = itertools.izip(targetSignal,distractorSignal)
		while not decided and n>=timeout:
			try:
				t,d = it.next()
			except (GeneratorExit,StopIteration):
				break
			n+=1
			ct+=t
			cd+=d
			post_va_t = 1./(1./self.prior_va_t**2+n/self.sigma_t**2)
			post_va_d = 1./(1./self.prior_va_d**2+n/self.sigma_d**2)
			post_mu_t = (self.prior_mu_t/self.prior_va_t**2+ct/self.sigma_t**2)*post_va_t
			post_mu_d = (self.prior_mu_d/self.prior_va_d**2+cd/self.sigma_d**2)*post_va_d
			
			dif_mu = post_mu_t-post_mu_d
			dif_si = np.sqrt(post_va_t+post_va_d)
			if abs(dif_mu)/dif_si>=self.threshold:
				decided = True
			if storeSignals:
				self.signals.append([t,d])
		if not decided:
			ret = decided,np.nan,np.nan,np.nan,n
		else:
			performance = 1 if dif_mu>0 else 0
			ret = decided,performance,dif_mu,dif_si,n
		return ret
	
	def batchInference(self,targetSignalArr,distractorSignalArr):
		if targetSignalArr.shape!=distractorSignalArr.shape:
			raise(ValueError("Target and distractor signal arrays must be numpy arrays with the same shape"))
		s = list(targetSignalArr.shape)
		s[1]+=1
		post_mu_t = self.prior_mu_t*np.ones(tuple(s))
		post_mu_d = self.prior_mu_d*np.ones(tuple(s))
		post_va_t = self.prior_va_t*np.ones(tuple(s))
		post_va_d = self.prior_va_d*np.ones(tuple(s))
		n = np.tile(np.reshape(np.arange(s[1]),(1,s[1])),(s[0],1))
		
		post_va_t = 1./(1./self.prior_va_t**2+n/self.sigma_t**2)
		post_va_d = 1./(1./self.prior_va_d**2+n/self.sigma_d**2)
		post_mu_t[:,1:] = (self.prior_mu_t/self.prior_va_t**2+np.cumsum(targetSignalArr,axis=1)/self.sigma_t**2)*post_va_t[:,1:]
		post_mu_d[:,1:] = (self.prior_mu_d/self.prior_va_d**2+np.cumsum(distractorSignalArr,axis=1)/self.sigma_d**2)*post_va_d[:,1:]
		
		dif_mu = post_mu_t-post_mu_d
		dif_si = np.sqrt(post_va_t+post_va_d)
		ret = np.zeros((s[0],5))
		for trial,(dm,ds) in enumerate(itertools.izip(dif_mu,dif_si)):
			dt = np.argmax(np.abs(dm)/ds>=self.threshold)
			if dt==s[1]-1:
				if np.abs(dm[-1])/ds[-1]<self.threshold:
					ret[trial] = np.array([False,np.nan,np.nan,np.nan,dt])
					continue
			performance = 1 if dm[dt]>0 else 0
			ret[trial] = np.array([True,performance,dm[dt],ds[dt],dt])
		return ret

class UnknownVarPerfectInference(PerfectInference):
	"""
	Class that implements perfect bayes inference on the mean luminance
	classification task with unknown model variance. The conjugate prior
	is a Gamma-Normal distribution that yields the Gaussiana mean (mu)
	and the precision (1/Var)
	"""
	def __init__(self,prior_mu_t=0.,prior_mu_d=0.,prior_nu_t=1.,prior_nu_d=1.,\
				prior_a_t=1.,prior_a_d=1.,prior_b_t=1.,prior_b_d=1.,threshold=5.):
		self.prior_mu_t = prior_mu_t
		self.prior_mu_d = prior_mu_d
		self.prior_nu_t = prior_nu_t
		self.prior_nu_d = prior_nu_d
		self.prior_a_t = prior_a_t
		self.prior_a_d = prior_a_d
		self.prior_b_t = prior_b_t
		self.prior_b_d = prior_b_d
		
		self.threshold = threshold
		self.signals = []
	
	def onlineInference(self,targetSignal,distractorSignal,storeSignals=False,timeout=np.inf):
		# The probability hyperparameters are updated with sample means
		# and variances. These must be computed online in a single pass.
		# Therefore we use Welford-Knuth algorithm.
		n = 0
		t_mv = onlineMeanVar()
		d_mv = onlineMeanVar()
		decided = False
		it = itertools.izip(targetSignal,distractorSignal)
		while not decided and n>=timeout:
			try:
				t,d = it.next()
			except (GeneratorExit,StopIteration):
				break
			n+=1
			t_mean,t_var = t_mv.add_value(t)
			d_mean,d_var = d_mv.add_value(d)
			
			# Target distribution hyperparameter update
			post_nu_t = self.prior_nu_t+n
			post_a_t =  self.prior_a_t+0.5*n
			post_mu_t = (self.prior_nu_t*self.prior_mu_t+n*t_mean)/(self.prior_nu_t+n)
			post_b_t =  self.prior_b_t+0.5*t_var+0.5*n*prior_nu_t*(t_mean-prior_mu_t)**2/(self.prior_nu_t+n)
			
			# Distractor distribution hyperparameter update
			post_nu_d = self.prior_nu_d+n
			post_a_d =  self.prior_a_d+0.5*n
			post_mu_d = (self.prior_nu_d*self.prior_mu_d+n*d_mean)/(self.prior_nu_d+n)
			post_b_d =  self.prior_b_d+0.5*d_var+0.5*n*prior_nu_d*(d_mean-prior_mu_d)**2/(self.prior_nu_d+n)
			
			# The distribution mean, mu, follows a student's t-distribution
			# with parameter 2 post_a. The distribution is scaled and
			# shifted, and is written in terms of variable x that reads
			# x = (mu-post_mu)**2*(post_nu*post_a/post_b)
			dif_mu = post_mu_t-post_mu_d
			dif_si = np.sqrt((post_b_t/post_nu_t/post_a_t)**2*post_nu_d/(post_nu_d-2)+(post_b_d/post_nu_d/post_a_d)**2*post_nu_d/(post_nu_d-2))
			if abs(dif_mu)/dif_si>=self.threshold:
				decided = True
			if storeSignals:
				self.signals.append([t,d])
		if decided:
			ret = decided,np.nan,np.nan,np.nan,n
		else:
			performance = 1 if dif_mu>0 else 0
			ret = decided,performance,dif_mu,dif_si,n
		return ret
	
	def batchInference(self,targetSignalArr,distractorSignalArr):
		if targetSignalArr.shape!=distractorSignalArr.shape:
			raise(ValueError("Target and distractor signal arrays must be numpy arrays with the same shape"))
		s = list(targetSignalArr.shape)
		s[1]+=1
		post_nu_t = self.prior_nu_t*np.ones(tuple(s))
		post_a_t =  self.prior_a_t*np.ones(tuple(s))
		post_mu_t = self.prior_mu_t*np.ones(tuple(s))
		post_b_t =  self.prior_b_t*np.ones(tuple(s))
		post_nu_d = self.prior_nu_d*np.ones(tuple(s))
		post_a_d =  self.prior_a_d*np.ones(tuple(s))
		post_mu_d = self.prior_mu_d*np.ones(tuple(s))
		post_b_d =  self.prior_b_d*np.ones(tuple(s))
		n = np.tile(np.reshape(np.arange(s[1]),(1,s[1])),(s[0],1))
		
		mv = onlineMeanVar()
		t_mean,t_var = mv.arr_add_value(targetSignalArr)
		mv.reset()
		d_mean,d_var = mv.arr_add_value(distractorSignalArr)
		
		# Target distribution hyperparameter update
		post_nu_t = self.prior_nu_t+n
		post_a_t =  self.prior_a_t+0.5*n
		post_mu_t[:,1:] = (self.prior_nu_t*self.prior_mu_t+n[:,1:]*t_mean)/(self.prior_nu_t+n[:,1:])
		post_b_t[:,1:] = self.prior_b_t+0.5*t_var+0.5*n[:,1:]*prior_nu_t*(t_mean-prior_mu_t)**2/(self.prior_nu_t+n[:,1:])
		
		# Distractor distribution hyperparameter update
		post_nu_d = self.prior_nu_d+n
		post_a_d =  self.prior_a_d+0.5*n
		post_mu_d[:,1:] = (self.prior_nu_d*self.prior_mu_d+n[:,1:]*d_mean)/(self.prior_nu_d+n[:,1:])
		post_b_d[:,1:] = self.prior_b_d+0.5*d_var+0.5*n[:,1:]*prior_nu_d*(d_mean-prior_mu_d)**2/(self.prior_nu_d+n[:,1:])
		
		
		# The distribution mean, mu, follows a student's t-distribution
		# with parameter 2 post_a. The distribution is scaled and
		# shifted, and is written in terms of variable x that reads
		# x = (mu-post_mu)**2*(post_nu*post_a/post_b)
		dif_mu = post_mu_t-post_mu_d
		dif_si = np.sqrt((post_b_t/post_nu_t/post_a_t)**2*post_nu_d/(post_nu_d-2)+(post_b_d/post_nu_d/post_a_d)**2*post_nu_d/(post_nu_d-2))
		ret = np.zeros((s[0],5))
		for trial,(dm,ds) in enumerate(itertools.izip(dif_mu,dif_si)):
			dt = np.argmax(np.abs(dm)/ds>=self.threshold)
			if dt==s[1]-1:
				if np.abs(dm[-1])/ds[-1]<self.threshold:
					ret[trial] = np.array([False,np.nan,np.nan,np.nan,dt])
					continue
			performance = 1 if dm[dt]>0 else 0
			ret[trial] = np.array([True,performance,dm[dt],ds[dt],dt])
		return ret

def fit_rt_distribution(subject,synthetic_trials=None,threshold=1.6,dead_time=400.,knownVariance=25.,priors=(0.,0.,1.,1.)):
	# Startup the model used to perform the inference depending on whether
	# variance is known or not
	if knownVariance:
		if not isinstance(priors,dict):
			model = KnownVarPerfectInference(np.sqrt(knownVariance),np.sqrt(knownVariance),*priors)
		else:
			model = KnownVarPerfectInference(np.sqrt(knownVariance),np.sqrt(knownVariance),**priors)
	else:
		if not isinstance(priors,dict):
			model = UnknownVarPerfectInference(*priors)
		else:
			model = UnknownVarPerfectInference(**priors)
	
	# The parameters that will be fitted are the decision threshold and
	# the dead time in ms after decision
	
	# Load the subject data and compute the subject's RT distribution
	dat,t,d = subject.load_data()
	max_rt_ind = int(math.ceil(max(dat[:,1])/40))
	bin_edges = np.array(range(max_rt_ind+1))*40
	subject_rt,_ = np.histogram(dat[:,1],bin_edges)
	subject_rt = subject_rt.astype(np.float)/float(dat.shape[0])
	
	# Prepare to compute the model RT distribution for a set of
	# syntethic trials
	if synthetic_trials:
		targetmean,indeces = io.increase_histogram_count(dat[:,0],synthetic_trials)
	else:
		targetmean = dat[:,0]
		indeces = np.array(range(dat.shape[0]),dtype=np.int)
	distractormean = 50
	sigma = 5.
	target = (np.random.randn(max_rt_ind,targetmean.shape[0])*sigma+targetmean).T
	distractor = (np.random.randn(max_rt_ind,targetmean.shape[0])*sigma+distractormean).T
	
	# Fitting initial guess
	x0 = np.array([threshold,dead_time])
	# Upper bound of parameter search space (lower bound is [0,0])
	ubound = np.array([20.,bin_edges[-1]])
	# Initial search width
	sigma0 = 1/3
	# Additional cma options
	cma_options = cma.CMAOptions()
	cma_options.set('bounds',[np.zeros(2),ubound])
	cma_options.set('scaling_of_variables',ubound)
	#~ return cma.fmin(merit,x0,sigma0,options=cma_options,args=(subject_rt,bin_edges,model,target,distractor),restarts=0),indeces,target,distractor,model
	return [[threshold,dead_time]],indeces,target,distractor,model

def merit(params,subject_rt,bin_edges,model,target,distractor):
	model.threshold = params[0]
	simulation = model.batchInference(target,distractor)
	simulated_rt,_ = np.histogram(simulation[:,4]+params[1],bin_edges)
	return np.sum((subject_rt-simulated_rt.astype(np.float)/float(target.shape[0]))**2)/(len(bin_edges)-1)

def test(data_dir='/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles'):
	try:
		from matplotlib import pyplot as plt
		loaded_plot_libs = True
	except:
		loaded_plot_libs = False
	# Load subject data
	subjects = io.unique_subjects(data_dir)
	ms = io.merge_subjects(subjects)
	
	# Fit model parameters
	fit_output,indeces,target,distractor,model = fit_rt_distribution(ms,10000)
	
	# Compute subject rt distribution (only for plotting purposes)
	dat,t,d = ms.load_data()
	ISI = 40.
	#~ max_rt_ind = int(math.ceil(max(dat[:,1])/ISI))
	#~ bin_edges = np.array(range(max_rt_ind+1))*ISI
	#~ subject_rt,bin_edges = np.histogram(dat[:,1],bin_edges)
	#~ subject_rt=subject_rt.astype(np.float)/float(dat.shape[0])
	# Compute subject kernels (only for plotting purposes)
	t = (np.mean(t,axis=2,keepdims=False).T-dat[:,0]).T
	d = np.mean(d,axis=2,keepdims=False)-50
	fluctuations = np.transpose(np.array([t,d]),(1,0,2))
	sdk,sck,sdk_std,sck_std = ke.kernels(fluctuations,1-dat[:,2],dat[:,3]-1)
	sT = np.array(range(sdk.shape[1]),dtype=float)*ISI
	
	#~ # Compute fitted model's rt distribution
	#~ model.threshold = fit_output[0][0]
	#~ simulation = model.batchInference(target,distractor)
	#~ simulated_rt,_ = np.histogram(simulation[:,4]+fit_output[0][1],bin_edges)
	#~ simulated_sel = 1-simulation[:,1]
	#~ simulated_sel[np.isnan(simulated_sel)] = 1
	#~ # Compute fitted model's kernels
	#~ np.array([target.T-dat[indeces,0],distractor-50]).shape
	#~ sim_fluctuations = np.transpose(np.array([target.T-dat[indeces,0],distractor.T-50]),(1,2,0))
	#~ sim_sdk,sim_sck,sim_sdk_std,sim_sck_std = ke.kernels(fluctuations,simulated_sel,np.ones_like(simulated_sel))
	#~ sim_sT = np.array(range(sim_sdk.shape[1]),dtype=float)*ISI
	
	#~ plt.figure(figsize=(13,10))
	#~ plt.show()
	#~ return 0
	#~ print bin_edges,subject_rt
	if loaded_plot_libs:
		plt.figure(figsize=(13,10))
		#~ plt.subplot(121)
		#~ plt.plot(bin_edges[:-1],subject_rt,'k')
		#~ plt.plot(bin_edges[:-1],simulated_rt,'r')
		#~ plt.xlabel('Response time [ms]')
		plt.subplot(122)
		plt.plot(sT,sdk[0],color='b',linestyle='--')
		plt.plot(sT,sdk[1],color='r',linestyle='--')
		#~ plt.plot(sim_sT,sim_sdk[0],color='b')
		#~ plt.plot(sim_sT,sim_sdk[1],color='r')
		#~ plt.fill_between(sT,sdk[0]-sdk_std[0],sdk[0]+sdk_std[0],color='b',alpha=0.3,edgecolor=None)
		#~ plt.fill_between(sT,sdk[1]-sdk_std[1],sdk[1]+sdk_std[1],color='r',alpha=0.3,edgecolor=None)
		#~ plt.fill_between(sim_sT,sim_sdk[0]-sim_sdk_std[0],sim_sdk[0]+sim_sdk_std[0],color='b',alpha=0.3,edgecolor=None)
		#~ plt.fill_between(sim_sT,sim_sdk[1]-sim_sdk_std[1],sim_sdk[1]+sim_sdk_std[1],color='r',alpha=0.3,edgecolor=None)
		#~ plt.plot(plt.gca().get_xlim(),[0,0],color='k')
		#~ plt.xlabel('Time [ms]')
		#~ plt.ylabel('Fluctuation [$cd/m^{2}$]')
		#~ plt.legend(['Subject $D_{S}$','Subject $D_{N}$','Simulated $D_{S}$','Simulated $D_{N}$'])
		plt.show()
	return 0

if __name__=="__main__":
	if len(sys.argv)>1:
		test(sys.argv[1])
	else:
		test()
