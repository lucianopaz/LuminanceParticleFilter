#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package that implements perfect inference behavior """

from __future__ import division
import numpy as np
import itertools, math

def find(a, predicate=lambda x: x, chunk_size=1024):
	"""
	Find the indices of array elements that match the predicate.
	
	Parameters
	----------
	a : array_like
		Input data, must be 1D.
	
	predicate : function
		A function which operates on sections of the given array, returning
		element-wise True or False for each data value.
	
	chunk_size : integer
		The length of the chunks to use when searching for matching indices.
		For high probability predicates, a smaller number will make this
		function quicker, similarly choose a larger number for low
		probabilities.
	
	Returns
	-------
	index_generator : generator
		A generator of (indices, data value) tuples which make the predicate
		True.
	
	See Also
	--------
	where, nonzero
	
	Notes
	-----
	This function is best used for finding the first, or first few, data values
	which match the predicate.
	
	Examples
	--------
	>>> a = np.sin(np.linspace(0, np.pi, 200))
	>>> result = find(a, lambda arr: arr > 0.9)
	>>> next(result)
	((71, ), 0.900479032457)
	>>> np.where(a > 0.9)[0][0]
	71
	"""
	if a.ndim != 1:
		raise ValueError('The array must be 1D, not {}.'.format(a.ndim))
	
	i0 = 0
	chunk_inds = itertools.chain(xrange(chunk_size, a.size, chunk_size), 
				 [None])
	
	for i1 in chunk_inds:
		chunk = a[i0:i1]
		for inds in itertools.izip(*predicate(chunk).nonzero()):
			yield inds[0] + i0, chunk[inds]
		i0 = i1

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

class KnownVarPerfectInference(PerfectInference):
	"""
	Class that implements perfect bayes inference on the mean luminance
	classification task with known model variance
	"""
	def __init__(self,model_var_t=1.,model_var_d=1.,prior_mu_t=0.,prior_mu_d=0.,prior_va_t=1.,prior_va_d=1.,\
				threshold=5.,ISI=1.):
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
		 - threshold:   Decision threshold
		 - ISI:         Inter stimulus interval (ISI). Represents the
		                time between changes of the target and distractor
		                stimuli.
		"""
		self.var_t = model_var_t
		self.var_d = model_var_d
		self.prior_mu_t = prior_mu_t
		self.prior_mu_d = prior_mu_d
		self.prior_va_t = prior_va_t
		self.prior_va_d = prior_va_d
		
		self.threshold = threshold
		self.ISI = ISI
		self.signals = []
	
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
			
			dif_mu = post_mu_t-post_mu_d
			dif_si = np.sqrt(post_va_t+post_va_d)
			if abs(dif_mu)/dif_si>=self.threshold:
				decided = True
			if storeSignals:
				self.signals.append([t,d])
		if not decided:
			ret = decided,np.nan,np.nan,np.nan,n*self.ISI
		else:
			performance = 1 if dif_mu>0 else 0
			ret = decided,performance,dif_mu,dif_si,n*self.ISI
		return ret
	
	def batchInference(self,targetSignalArr,distractorSignalArr,returnCriteria=False):
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
		
		dif_mu = post_mu_t-post_mu_d
		dif_si = np.sqrt(post_va_t+post_va_d)
		ret = np.zeros((s[0],5))
		for trial,(dm,ds) in enumerate(itertools.izip(dif_mu,dif_si)):
			threshold_crossed = find(np.abs(dm)/ds>=self.threshold)
			try:
				dt,_ = next(threshold_crossed)
				performance = 1 if dm[dt]>0 else 0
				ret[trial] = np.array([True,performance,dm[dt],ds[dt],dt*self.ISI])
			except:
				ret[trial] = np.array([False,np.nan,np.nan,np.nan,(s[1]-1)*self.ISI])
		if returnCriteria:
			ret = (ret,(post_mu_t-post_mu_d)/np.sqrt(post_va_t+post_va_d))
		return ret

class UnknownVarPerfectInference(PerfectInference):
	"""
	Class that implements perfect bayes inference on the mean luminance
	classification task with unknown model variance. The conjugate prior
	is a Gamma-Normal distribution that yields the Gaussiana mean (mu)
	and the precision (1/Var)
	"""
	def __init__(self,prior_mu_t=0.,prior_mu_d=0.,prior_nu_t=1.,prior_nu_d=1.,\
				prior_a_t=1.,prior_a_d=1.,prior_b_t=1.,prior_b_d=1.,threshold=5.,ISI=1.):
		"""
		Unknown variance perfect inference constructor
		Syntax:
		instance = UnknownVarPerfectInference(prior_mu_t=0.,prior_mu_d=0.,prior_nu_t=1.,prior_nu_d=1.,\
				prior_a_t=1.,prior_a_d=1.,prior_b_t=1.,prior_b_d=1.,threshold=5.,ISI=1.)
		
		Input:
		 - prior_mu_t:  Mean of the target prior distribution
		 - prior_mu_d:  Mean of the distractor prior distribution
		 - prior_nu_t:  Nu (sometime lambda) of the target prior distribution
		 - prior_nu_d:  Nu (sometime lambda) of the distractor prior distribution
		 - prior_a_t:   Alpha of the target prior distribution
		 - prior_a_d:   Alpha of the distractor prior distribution
		 - prior_b_t:   Beta of the target prior distribution
		 - prior_b_d:   Beta of the distractor prior distribution
		 - threshold:   Decision threshold
		 - ISI:         Inter stimulus interval (ISI). Represents the
		                time between changes of the target and distractor
		                stimuli.
		"""
		self.prior_mu_t = prior_mu_t
		self.prior_mu_d = prior_mu_d
		self.prior_nu_t = prior_nu_t
		self.prior_nu_d = prior_nu_d
		self.prior_a_t = prior_a_t
		self.prior_a_d = prior_a_d
		self.prior_b_t = prior_b_t
		self.prior_b_d = prior_b_d
		
		self.ISI = ISI
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
			
			# The distribution mean, mu, follows a student's t-distribution
			# with parameter 2 post_a. The distribution is scaled and
			# shifted, and is written in terms of variable x that reads
			# x = (mu-post_mu)**2*(post_nu*post_a/post_b)
			dif_mu = post_mu_t-post_mu_d
			if n>2:
				dif_si = np.sqrt((post_b_t/post_nu_t/post_a_t)**2*post_nu_d/(post_nu_d-2)+(post_b_d/post_nu_d/post_a_d)**2*post_nu_d/(post_nu_d-2))
			else:
				dif_si = np.inf
			if abs(dif_mu)/dif_si>=self.threshold:
				decided = True
			if storeSignals:
				self.signals.append([t,d])
		if not decided:
			ret = decided,np.nan,np.nan,np.nan,n*self.ISI
		else:
			performance = 1 if dif_mu>0 else 0
			ret = decided,performance,dif_mu,dif_si,n*self.ISI
		return ret
	
	def batchInference(self,targetSignalArr,distractorSignalArr,returnCriteria=False):
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
		
		
		# The distribution mean, mu, follows a student's t-distribution
		# with parameter 2 post_a. The distribution is scaled and
		# shifted, and is written in terms of variable x that reads
		# x = (mu-post_mu)**2*(post_nu*post_a/post_b)
		dif_mu = post_mu_t-post_mu_d
		dif_si = np.inf*np.ones_like(dif_mu)
		dif_si[:,2:] = np.sqrt((post_b_t[:,2:]/post_nu_t[:,2:]/post_a_t[:,2:])**2*post_nu_d[:,2:]/(post_nu_d[:,2:]-2)+(post_b_d[:,2:]/post_nu_d[:,2:]/post_a_d[:,2:])**2*post_nu_d[:,2:]/(post_nu_d[:,2:]-2))
		ret = np.zeros((s[0],5))
		for trial,(dm,ds) in enumerate(itertools.izip(dif_mu,dif_si)):
			threshold_crossed = find(np.abs(dm)/ds>=self.threshold)
			try:
				dt,_ = next(threshold_crossed)
				performance = 1 if dm[dt]>0 else 0
				ret[trial] = np.array([True,performance,dm[dt],ds[dt],dt*self.ISI])
			except:
				ret[trial] = np.array([False,np.nan,np.nan,np.nan,(s[1]-1)*self.ISI])
		if returnCriteria:
			ret = (ret,(post_mu_t-post_mu_d)/np.sqrt(post_va_t+post_va_d))
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
	onlinekRes = np.zeros((trials,5))
	onlineuRes = np.zeros((trials,5))
	for trial,(t,d) in enumerate(itertools.izip(target,distractor)):
		# Known variance inference
		decided,performance,dif_mu,dif_si,rt = kmodel.onlineInference(t,d)
		onlinekRes[trial] = np.array([decided,performance,dif_mu,dif_si,rt])
		# Unknown variance inference
		decided,performance,dif_mu,dif_si,rt = umodel.onlineInference(t,d)
		onlineuRes[trial] = np.array([decided,performance,dif_mu,dif_si,rt])
	batchkRes = kmodel.batchInference(target,distractor)
	batchuRes = umodel.batchInference(target,distractor)
	
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

if __name__=="__main__":
	test()
