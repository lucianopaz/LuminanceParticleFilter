#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package that implements perfect inference behavior """

from __future__ import division
import numpy as np
import itertools, math, copy

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
		self.var = self.M2/(self.n-1) if self.n>1 else (0. if not isinstance(self.mean,np.ndarray) else np.zeros_like(self.mean))
	def val(self):
		return (self.mean,self.var)
	def add_value(self,value):
		self.update(value)
		return self.val()
	def arr_add_value(self,val_arr,axis=0):
		mean = np.zeros_like(val_arr)
		var =  np.zeros_like(val_arr)
		indeces = [range(l) for l in list(val_arr.shape)]
		ix = indeces
		for i in indeces[axis]:
			ix[axis] = [i]
			mean[np.ix_(*ix)],var[np.ix_(*ix)] = self.add_value(val_arr[np.ix_(*ix)])
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
	def set_symmetric_threshold(self,threshold):
		self.upper_threshold = copy.copy(threshold)
		self.lower_threshold = copy.copy(-threshold)
	def set_asymmetric_threshold(self,thresholds):
		self.upper_threshold = copy.copy(thresholds[0])
		self.lower_threshold = copy.copy(thresholds[1])

class KnownVarPerfectInference(PerfectInference):
	"""
	Class that implements perfect bayes inference on the mean luminance
	classification task with known model variance
	"""
	def __init__(self,model_var_t=1.,model_var_d=1.,prior_mu_t=0.,prior_mu_d=0.,prior_va_t=1.,prior_va_d=1.,\
				lower_threshold=-5.,upper_threshold=5.,ISI=1.):
		"""
		Known variance perfect inference constructor
		Syntax:
		instance = KnownVarPerfectInference(model_var_t=1.,model_var_d=1.,prior_mu_t=0.,prior_mu_d=0.,prior_va_t=1.,prior_va_d=1.,\
				threshold=5.,ISI=1.)
		
		Input:
		 - model_var_t: Known variance of the target stimulus
		 - model_var_d: Known variance of the distractor stimulus
		 - prior_mu_t:  Mean of the target prior distribution
		 - prior_mu_d:  Mean of the distractor prior distribution
		 - prior_va_t:  Variance of the target prior distribution
		 - prior_va_d:  Variance of the distractor prior distribution
		 - lower_threshold:   Decision threshold for selecting distractor
		 - upper_threshold:   Decision threshold for selecting target
		 - ISI:         Inter stimulus interval (ISI). Represents the
		                time between changes of the target and distractor
		                stimuli.
		"""
		self.var_t = float(model_var_t)
		self.var_d = float(model_var_d)
		self.prior_mu_t = float(prior_mu_t)
		self.prior_mu_d = float(prior_mu_d)
		self.prior_va_t = float(prior_va_t)
		self.prior_va_d = float(prior_va_d)
		
		self.lower_threshold = float(lower_threshold)
		self.upper_threshold = float(upper_threshold)
		self.ISI = float(ISI)
		self.signals = []
	
	def criteria(self,post_mu_t,post_mu_d,post_va_t,post_va_d):
		"""
		Optimal decision criteria for discriminating which of two gaussians
		has higher mean. (mu_t-mu_d)/sqrt(var_t+var_d)
		"""
		return (post_mu_t-post_mu_d)/np.sqrt(post_va_t+post_va_d)
	
	def copy(self):
		output = KnownVarPerfectInference(model_var_t=self.var_t,model_var_d=self.var_d,\
				prior_mu_t=self.prior_mu_t,prior_mu_d=self.prior_mu_d,prior_va_t=self.prior_va_t,\
				prior_va_d=self.prior_va_d,lower_threshold=self.lower_threshold,\
				upper_threshold=self.upper_threshold,ISI=self.ISI)
		output.signals = self.signals.copy()
		return output
	
	def onlineInference(self,targetSignal,distractorSignal,storeSignals=False,timeout=np.inf):
		n = 0
		ct = 0.
		cd = 0.
		decided = False
		it = itertools.izip(targetSignal,distractorSignal)
		while not decided and n<timeout:
			try:
				t,d = it.next()
			except (GeneratorExit,StopIteration):
				break
			n+=1
			ct+=t
			cd+=d
			post_va_t = 1./(1./self.prior_va_t+n/self.var_t)
			post_va_d = 1./(1./self.prior_va_d+n/self.var_d)
			post_mu_t = (self.prior_mu_t/self.prior_va_t+ct/self.var_t)*post_va_t
			post_mu_d = (self.prior_mu_d/self.prior_va_d+cd/self.var_d)*post_va_d
			
			criterium = self.criteria(post_mu_t,post_mu_d,post_va_t,post_va_d)
			if criterium>=self.upper_threshold:
				decided = True
				performance = 1
			elif criterium<=self.lower_threshold:
				decided = True
				performance = 0
			if storeSignals:
				self.signals.append([t,d])
		if not decided:
			ret = decided,np.nan,np.nan,np.nan,n*self.ISI
		else:
			ret = decided,performance,criterium,n*self.ISI
		return ret
	
	def passiveInference(self,targetSignalArr,distractorSignalArr):
		if targetSignalArr.shape!=distractorSignalArr.shape:
			raise(ValueError("Target and distractor signal arrays must be numpy arrays with the same shape"))
		s = list(targetSignalArr.shape)
		s[1]+=1
		post_mu_t = self.prior_mu_t*np.ones(tuple(s))
		post_mu_d = self.prior_mu_d*np.ones(tuple(s))
		post_va_t = self.prior_va_t*np.ones(tuple(s))
		post_va_d = self.prior_va_d*np.ones(tuple(s))
		n = np.tile(np.reshape(np.arange(s[1]),(1,s[1])),(s[0],1))
		
		post_va_t = 1./(1./self.prior_va_t+n/self.var_t)
		post_va_d = 1./(1./self.prior_va_d+n/self.var_d)
		post_mu_t[:,1:] = (self.prior_mu_t/self.prior_va_t+np.cumsum(targetSignalArr,axis=1)/self.var_t)*post_va_t[:,1:]
		post_mu_d[:,1:] = (self.prior_mu_d/self.prior_va_d+np.cumsum(distractorSignalArr,axis=1)/self.var_d)*post_va_d[:,1:]
		return post_mu_t,post_mu_d,post_va_t,post_va_d
	
	def batchInference(self,targetSignalArr,distractorSignalArr,returnPosterior=False,returnCriteria=False):
		if targetSignalArr.shape!=distractorSignalArr.shape:
			raise(ValueError("Target and distractor signal arrays must be numpy arrays with the same shape"))
		if isinstance(self.upper_threshold,np.ndarray):
			upper_threshold = self.upper_threshold[:(targetSignalArr.shape[-1]+1)]
			lower_threshold = self.lower_threshold[:(targetSignalArr.shape[-1]+1)]
		else:
			upper_threshold = self.upper_threshold
			lower_threshold = self.lower_threshold
		s = list(targetSignalArr.shape)
		s[1]+=1
		post_mu_t = self.prior_mu_t*np.ones(tuple(s))
		post_mu_d = self.prior_mu_d*np.ones(tuple(s))
		post_va_t = self.prior_va_t*np.ones(tuple(s))
		post_va_d = self.prior_va_d*np.ones(tuple(s))
		n = np.tile(np.reshape(np.arange(s[1]),(1,s[1])),(s[0],1))
		
		post_va_t = 1./(1./self.prior_va_t+n/self.var_t)
		post_va_d = 1./(1./self.prior_va_d+n/self.var_d)
		post_mu_t[:,1:] = (self.prior_mu_t/self.prior_va_t+np.cumsum(targetSignalArr,axis=1)/self.var_t)*post_va_t[:,1:]
		post_mu_d[:,1:] = (self.prior_mu_d/self.prior_va_d+np.cumsum(distractorSignalArr,axis=1)/self.var_d)*post_va_d[:,1:]
		
		criterium = self.criteria(post_mu_t,post_mu_d,post_va_t,post_va_d)
		ret = np.zeros((s[0],4))
		if returnPosterior:
			posterior = {'mu_t':np.nan*np.zeros(s[0]),'mu_d':np.nan*np.zeros(s[0]),'var_t':np.nan*np.zeros(s[0]),'var_d':np.nan*np.zeros(s[0])}
		for trial,cr in enumerate(criterium):
			dt1 = np.flatnonzero(cr>=upper_threshold)
			dt2 = np.flatnonzero(cr<=lower_threshold)
			if dt1.size==0 and dt2.size==0:
				decided = False
				performance = np.nan
				crit = np.nan
				RT = (s[1]-1)*self.ISI
			else:
				decided = True
				if dt1.size!=0 and dt2.size==0:
					performance = 1
					dt = dt1[0]
				elif dt1.size==0 and dt2.size!=0:
					performance = 0
					dt = dt2[0]
				else:
					if dt1[0]<dt2[0]:
						performance = 1
						dt = dt1[0]
					else:
						performance = 0
						dt = dt2[0]
				crit = cr[dt]
				RT = dt*self.ISI
				if returnPosterior:
					posterior['mu_t'][trial] = post_mu_t[trial,dt]
					posterior['mu_d'][trial] = post_mu_d[trial,dt]
					posterior['var_t'][trial] = post_va_t[trial,dt]
					posterior['var_d'][trial] = post_va_d[trial,dt]
			ret[trial] = np.array([decided,performance,crit,RT])
		ret = (ret,)
		if returnPosterior:
			ret+= (posterior,)
		if returnCriteria:
			ret+= (criterium,)
		if len(ret)==1:
			ret = ret[0]
		return ret

class UnknownVarPerfectInference(PerfectInference):
	"""
	Class that implements perfect bayes inference on the mean luminance
	classification task with unknown model variance. The conjugate prior
	is a Gamma-Normal distribution that yields the Gaussiana mean (mu)
	and the precision (1/Var)
	"""
	def __init__(self,prior_mu_t=0.,prior_mu_d=0.,prior_nu_t=1.,prior_nu_d=1.,\
				prior_a_t=1.,prior_a_d=1.,prior_b_t=1.,prior_b_d=1.,lower_threshold=-5.,upper_threshold=5.,ISI=1.):
		"""
		Unknown variance perfect inference constructor
		Syntax:
		instance = UnknownVarPerfectInference(prior_mu_t=0.,prior_mu_d=0.,prior_nu_t=1.,prior_nu_d=1.,\
				prior_a_t=1.,prior_a_d=1.,prior_b_t=1.,prior_b_d=1.,threshold=5.,ISI=1.)
		
		Input:
		 - prior_mu_t:        Mean of the target prior distribution
		 - prior_mu_d:        Mean of the distractor prior distribution
		 - prior_nu_t:        Nu (sometime lambda) of the target prior distribution
		 - prior_nu_d:        Nu (sometime lambda) of the distractor prior distribution
		 - prior_a_t:         Alpha of the target prior distribution
		 - prior_a_d:         Alpha of the distractor prior distribution
		 - prior_b_t:         Beta of the target prior distribution
		 - prior_b_d:         Beta of the distractor prior distribution
		 - lower_threshold:   Decision threshold for selecting distractor
		 - upper_threshold:   Decision threshold for selecting target
		 - ISI:               Inter stimulus interval (ISI). Represents
		                      the time between changes of the target and
		                      distractor stimuli.
		"""
		self.prior_mu_t = float(prior_mu_t)
		self.prior_mu_d = float(prior_mu_d)
		self.prior_nu_t = float(prior_nu_t)
		self.prior_nu_d = float(prior_nu_d)
		self.prior_a_t = float(prior_a_t)
		self.prior_a_d = float(prior_a_d)
		self.prior_b_t = float(prior_b_t)
		self.prior_b_d = float(prior_b_d)
		
		self.ISI = float(ISI)
		self.upper_threshold = float(upper_threshold)
		self.lower_threshold = float(lower_threshold)
		self.signals = []
	
	def criteria(self,post_mu_t,post_mu_d,post_nu_t,post_nu_d,post_a_t,post_a_d,post_b_t,post_b_d):
		# The distribution mean, mu, follows a student's t-distribution
		# with parameter 2 post_a. The distribution is scaled and
		# shifted, and is written in terms of variable x that reads
		# x = (mu-post_mu)**2*(post_nu*post_a/post_b)
		dif_mu = post_mu_t-post_mu_d
		if isinstance(dif_mu,np.ndarray):
			dif_si = np.inf*np.ones_like(dif_mu)
			dif_si[:,2:] = np.sqrt((post_b_t[:,2:]/post_nu_t[:,2:]/post_a_t[:,2:])**2*post_nu_d[:,2:]/(post_nu_d[:,2:]-2)+(post_b_d[:,2:]/post_nu_d[:,2:]/post_a_d[:,2:])**2*post_nu_d[:,2:]/(post_nu_d[:,2:]-2))
		else:
			if post_nu_d>2:
				dif_si = np.sqrt((post_b_t/post_nu_t/post_a_t)**2*post_nu_d/(post_nu_d-2)+(post_b_d/post_nu_d/post_a_d)**2*post_nu_d/(post_nu_d-2))
			else:
				dif_si = np.inf
		return dif_mu/dif_si
	
	def copy(self):
		self.prior_mu_t = prior_mu_t
		self.prior_mu_d = prior_mu_d
		self.prior_nu_t = prior_nu_t
		self.prior_nu_d = prior_nu_d
		self.prior_a_t = prior_a_t
		self.prior_a_d = prior_a_d
		self.prior_b_t = prior_b_t
		self.prior_b_d = prior_b_d
		output = KnownVarPerfectInference(prior_mu_t=self.prior_mu_t,prior_mu_d=self.prior_mu_d,\
				prior_nu_t=self.prior_nu_t,prior_nu_d=self.prior_nu_d,prior_a_t=self.prior_a_t,\
				prior_a_d=self.prior_a_d,prior_b_t=self.prior_b_t,prior_b_d=self.prior_b_d,\
				lower_threshold=self.lower_threshold,upper_threshold=self.upper_threshold,ISI=self.ISI)
		output.signals = self.signals.copy()
		return output
	
	def onlineInference(self,targetSignal,distractorSignal,storeSignals=False,timeout=np.inf):
		# The probability hyperparameters are updated with sample means
		# and variances. These must be computed online in a single pass.
		# Therefore we use Welford-Knuth algorithm.
		n = 0
		t_mv = onlineMeanVar()
		d_mv = onlineMeanVar()
		decided = False
		it = itertools.izip(targetSignal,distractorSignal)
		while not decided and n<timeout:
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
			post_b_t =  self.prior_b_t+0.5*t_var+0.5*n*self.prior_nu_t*(t_mean-self.prior_mu_t)**2/(self.prior_nu_t+n)
			
			# Distractor distribution hyperparameter update
			post_nu_d = self.prior_nu_d+n
			post_a_d =  self.prior_a_d+0.5*n
			post_mu_d = (self.prior_nu_d*self.prior_mu_d+n*d_mean)/(self.prior_nu_d+n)
			post_b_d =  self.prior_b_d+0.5*d_var+0.5*n*self.prior_nu_d*(d_mean-self.prior_mu_d)**2/(self.prior_nu_d+n)
			
			criterium = self.criteria(post_mu_t,post_mu_d,post_nu_t,post_nu_d,post_a_t,post_a_d,post_b_t,post_b_d)
			if criterium>=self.upper_threshold:
				decided = True
				performance = 1
			elif criterium<=self.lower_threshold:
				decided = True
				performance = 0
			if storeSignals:
				self.signals.append([t,d])
		if not decided:
			ret = decided,np.nan,np.nan,n*self.ISI
		else:
			ret = decided,performance,criterium,n*self.ISI
		return ret
	
	def passiveInference(self,targetSignalArr,distractorSignalArr):
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
		t_mean,t_var = mv.arr_add_value(targetSignalArr,axis=1)
		mv.reset()
		d_mean,d_var = mv.arr_add_value(distractorSignalArr,axis=1)
		
		# Target distribution hyperparameter update
		post_nu_t = self.prior_nu_t+n
		post_a_t =  self.prior_a_t+0.5*n
		post_mu_t[:,1:] = (self.prior_nu_t*self.prior_mu_t+n[:,1:]*t_mean)/(self.prior_nu_t+n[:,1:])
		post_b_t[:,1:] = self.prior_b_t+0.5*t_var+0.5*n[:,1:]*self.prior_nu_t*(t_mean-self.prior_mu_t)**2/(self.prior_nu_t+n[:,1:])
		
		# Distractor distribution hyperparameter update
		post_nu_d = self.prior_nu_d+n
		post_a_d =  self.prior_a_d+0.5*n
		post_mu_d[:,1:] = (self.prior_nu_d*self.prior_mu_d+n[:,1:]*d_mean)/(self.prior_nu_d+n[:,1:])
		post_b_d[:,1:] = self.prior_b_d+0.5*d_var+0.5*n[:,1:]*self.prior_nu_d*(d_mean-self.prior_mu_d)**2/(self.prior_nu_d+n[:,1:])
		
		return post_mu_t,post_mu_d,post_nu_t,post_nu_d,post_a_t,post_a_d,post_b_t,post_b_t
	
	def batchInference(self,targetSignalArr,distractorSignalArr,returnPosterior=False,returnCriteria=False):
		if targetSignalArr.shape!=distractorSignalArr.shape:
			raise(ValueError("Target and distractor signal arrays must be numpy arrays with the same shape"))
		if isinstance(self.upper_threshold,np.ndarray):
			upper_threshold = self.upper_threshold[:(targetSignalArr.shape[-1]+1)]
			lower_threshold = self.lower_threshold[:(targetSignalArr.shape[-1]+1)]
		else:
			upper_threshold = self.upper_threshold
			lower_threshold = self.lower_threshold
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
		t_mean,t_var = mv.arr_add_value(targetSignalArr,axis=1)
		mv.reset()
		d_mean,d_var = mv.arr_add_value(distractorSignalArr,axis=1)
		
		# Target distribution hyperparameter update
		post_nu_t = self.prior_nu_t+n
		post_a_t =  self.prior_a_t+0.5*n
		post_mu_t[:,1:] = (self.prior_nu_t*self.prior_mu_t+n[:,1:]*t_mean)/(self.prior_nu_t+n[:,1:])
		post_b_t[:,1:] = self.prior_b_t+0.5*t_var+0.5*n[:,1:]*self.prior_nu_t*(t_mean-self.prior_mu_t)**2/(self.prior_nu_t+n[:,1:])
		
		# Distractor distribution hyperparameter update
		post_nu_d = self.prior_nu_d+n
		post_a_d =  self.prior_a_d+0.5*n
		post_mu_d[:,1:] = (self.prior_nu_d*self.prior_mu_d+n[:,1:]*d_mean)/(self.prior_nu_d+n[:,1:])
		post_b_d[:,1:] = self.prior_b_d+0.5*d_var+0.5*n[:,1:]*self.prior_nu_d*(d_mean-self.prior_mu_d)**2/(self.prior_nu_d+n[:,1:])
		
		criterium = self.criteria(post_mu_t,post_mu_d,post_nu_t,post_nu_d,post_a_t,post_a_d,post_b_t,post_b_d)
		ret = np.zeros((s[0],4))
		if returnPosterior:
			posterior = {'mu_t':np.nan*np.zeros(s[0]),'mu_d':np.nan*np.zeros(s[0]),'nu_t':np.nan*np.zeros(s[0]),'nu_d':np.nan*np.zeros(s[0]),'b_t':np.nan*np.zeros(s[0]),'b_d':np.nan*np.zeros(s[0]),'a_t':np.nan*np.zeros(s[0]),'a_d':np.nan*np.zeros(s[0])}
		for trial,cr in enumerate(criterium):
			dt1 = np.flatnonzero(cr>=upper_threshold)
			dt2 = np.flatnonzero(cr<=lower_threshold)
			if dt1.size==0 and dt2.size==0:
				decided = False
				performance = np.nan
				crit = np.nan
				RT = (s[1]-1)*self.ISI
			else:
				decided = True
				if dt1.size!=0 and dt2.size==0:
					performance = 1
					dt = dt1[0]
					crit = cr[dt]
					RT = dt1[0]*self.ISI
				elif dt1.size==0 and dt2.size!=0:
					performance = 0
					dt = dt2[0]
					crit = cr[dt]
					RT = dt*self.ISI
				else:
					if dt1[0]<dt2[0]:
						dt = dt1[0]
						performance = 1
						crit = cr[dt]
						RT = dt*self.ISI
					else:
						dt = dt2[0]
						performance = 0
						crit = cr[dt]
						RT = dt*self.ISI
				if returnPosterior:
					posterior['mu_t'][trial] = post_mu_t[trial,dt]
					posterior['mu_d'][trial] = post_mu_d[trial,dt]
					posterior['nu_t'][trial] = post_nu_t[trial,dt]
					posterior['nu_d'][trial] = post_nu_d[trial,dt]
					posterior['a_t'][trial] = post_a_t[trial,dt]
					posterior['a_d'][trial] = post_a_t[trial,dt]
					posterior['b_t'][trial] = post_b_t[trial,dt]
					posterior['b_d'][trial] = post_b_t[trial,dt]
			ret[trial] = np.array([decided,performance,crit,RT])
		ret = (ret,)
		if returnPosterior:
			ret+= (posterior,)
		if returnCriteria:
			ret+= (criterium,)
		if len(ret)==1:
			ret = ret[0]
		return ret

def test():
	var_t = 0.3**2
	var_d = 0.5**2
	mean_t = 1.
	mean_d = 0.7
	stimlen = 1000
	trials = 3
	target = np.random.randn(trials,stimlen)*np.sqrt(var_t)+mean_t
	distractor = np.random.randn(trials,stimlen)*np.sqrt(var_d)+mean_d
	
	kmodel = KnownVarPerfectInference(model_var_t=var_t,model_var_d=var_d)
	umodel = UnknownVarPerfectInference()
	onlinekRes = np.zeros((trials,4))
	onlineuRes = np.zeros((trials,4))
	for trial,(t,d) in enumerate(itertools.izip(target,distractor)):
		# Known variance inference
		decided,performance,criter,rt = kmodel.onlineInference(t,d)
		onlinekRes[trial] = np.array([decided,performance,criter,rt])
		# Unknown variance inference
		decided,performance,criter,rt = umodel.onlineInference(t,d)
		onlineuRes[trial] = np.array([decided,performance,criter,rt])
	batchkRes,kposterior = kmodel.batchInference(target,distractor,returnPosterior=True)
	batchuRes,uposterior = umodel.batchInference(target,distractor,returnPosterior=True)
	
	kequal = False
	if np.all(np.isnan(onlinekRes)==np.isnan(onlinekRes)):
		if np.all(onlinekRes[np.logical_not(np.isnan(onlinekRes))]==batchkRes[np.logical_not(np.isnan(batchkRes))]):
			kequal = True
	uequal = False
	if np.all(np.isnan(onlineuRes)==np.isnan(onlineuRes)):
		if np.all(onlineuRes[np.logical_not(np.isnan(onlineuRes))]==batchuRes[np.logical_not(np.isnan(batchuRes))]):
			uequal = True
	
	print "Online and batch comparison"
	print "Known variance comparison = ",kequal
	print "Unknown variance comparison = ",uequal
	print "Known variance results"
	print onlinekRes
	if not kequal:
		print batchkRes
	print "Unknown variance results"
	print onlineuRes
	if not uequal:
		print batchuRes
	print "Known variance posteriors"
	print kposterior
	print "Unknown variance posteriors"
	print uposterior
	
	matlab_comparison()

def matlab_comparison():
	from scipy import io as io
	from matplotlib import pyplot as plt
	import kernels as ke
	aux = io.loadmat('mat_py_comp.mat')
	model = KnownVarPerfectInference(model_var_t=aux['sigma'][0][0]**2,model_var_d=aux['sigma'][0][0]**2,\
			prior_mu_t=aux['pr_mu_t'][0][0],prior_mu_d=aux['pr_mu_d'][0][0],prior_va_t=aux['pr_va_t'][0][0],prior_va_d=aux['pr_va_d'][0][0],\
			ISI=aux['ISI'][0][0])
	model.set_symmetric_threshold(aux['threshold'][0][0])
	target = aux['target']
	distractor = aux['distractor']
	mat_ret = aux['ret']
	
	#~ def dprime_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
		#~ return post_mu_t/np.sqrt(post_va_t)-post_mu_d/np.sqrt(post_va_d)
	#~ 
	#~ def dprime_var_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
		#~ return post_mu_t/post_va_t-post_mu_d/post_va_d
	#~ model.criteria = dprime_criteria
	py_ret,criterium = model.batchInference(target,distractor,returnCriteria=True)
	equal = False
	if np.all(np.isnan(mat_ret)==np.isnan(py_ret)):
		mat_temp = mat_ret[np.logical_not(np.isnan(mat_ret))]
		py_temp = py_ret[np.logical_not(np.isnan(py_ret))]
		if np.all(np.abs(py_temp-mat_temp)<1e-12):
			equal = True
	print "Got equal results in python and matlab? ",equal
	
	fluctuations = np.transpose(np.array([aux['tfluct'],aux['dfluct']]),(1,0,2))
	selection = 1-py_ret[:,1]
	#~ selection[np.isnan(selection)] = 1
	py_dk,_,py_dks,_ = ke.kernels(fluctuations,selection,np.ones_like(selection))
	
	print "Got equal kernels in python and matlab? ",True if (np.sum(np.abs(py_dk-aux['dk']))<1e-12 and np.sum(np.abs(py_dks-aux['dks']))<1e-12) else False
	
	plt.figure
	plt.subplot(211)
	plt.imshow(criterium[:,1:]-aux['dprime'],aspect='auto',interpolation='none')
	plt.title("Difference between the criteria for every trial and time step")
	plt.colorbar()
	plt.subplot(212)
	plt.imshow(py_ret-mat_ret,aspect='auto',interpolation='none')
	plt.title("Difference between the python and matlab result matrices for every trial")
	plt.colorbar()
	
	plt.figure()
	plt.subplot(211)
	plt.plot(py_dk.T-aux['dk'].T)
	plt.title('Decision kernel difference')
	plt.subplot(212)
	plt.plot(py_dks.T-aux['dks'].T)
	plt.title('Decision kernel std difference')
	plt.show()

if __name__=="__main__":
	test()
