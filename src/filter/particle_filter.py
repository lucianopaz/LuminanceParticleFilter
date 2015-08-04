#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package defining a basic particle filter """

from __future__ import division
import numpy as np
import math, random, scipy, copy, itertools

class Particle:
	"""
	Particle class has two properties, its weight (the log of the
	weight is also stored) and its value. The value is the random
	variable obtained when sampling from a Markov Chain
	"""
	def __init__(self,value=None,weight=0):
		self.value = value
		self.weight = weight
		self.is_vectorizable = isinstance(value,np.ndarray)
	def copy(self):
		return Particle(copy.copy(self.value),self.weight)

class Distribution(object):
	"""
	Virtual base class Distribution that only provides a method to
	sample from it
	"""
	is_vectorizable = False
	def sample(self):
		return Particle()

class VectDistribution(Distribution):
	"""
	Virtual base class VectDistribution that only provides a method to
	sample from it in a vectorial manner
	"""
	is_vectorizable = True
	def vectSample(self,*args):
		return (None,None)

class ParticleSet:
	"""
	ParticleSet class works as a special particle container that
	stores the number of particles, the particle objects and a list
	of weights for faster access
	"""
	def __init__(self,N=1,distribution=Distribution()):
		self.N = int(N)
		self.weights = np.array([1./self.N for _ in range(N)])
		if distribution.is_vectorizable:
			self.v_val = distribution.vectSample(N)
			self.is_vectorizable = True
			self.particlelize()
		else:
			self.particles = [distribution.sample() for _ in range(N)]
			for i,part in enumerate(self.particles):
				part.weight = self.weights[i]
			self.is_vectorizable = all([p.is_vectorizable for p in self.particles])
			if self.is_vectorizable:
				self.vectorize()
	def append(self,part):
		self.N+=1
		self.weights.append(part.weight)
		self.particles.append(part)
		if self.is_vectorizable:
			self.vectorize()
	def pop(self,part,index=-1):
		self.N-=1
		self.weigths.pop(index)
		ret =  self.particles.pop(index)
		if self.is_vectorizable:
			self.vectorize()
		return ret
	def copy(self):
		copied_set = ParticleSet(N=0)
		copied_set.N = self.N
		copied_set.weights = copy.copy(self.weights)
		copied_set.particles = [part.copy() for part in self.particles]
		copied_set.is_vectorizable = self.is_vectorizable
		if copied_set.is_vectorizable:
			copied_set.vectorize()
		return copied_set
	def replace(self,index,particle):
		self.particles[index] = particle
		self.weights[index] = particle.weight
		if self.is_vectorizable:
			self.v_val[index] = particle.value
	def __getitem__(self,index):
		if index>=self.N or index<-self.N:
			raise(IndexError)
		return self.particles[index]
	def effective_sample_size(self):
		# Kish approximate sample size formula
		return np.sum(self.weights)**2/np.sum(self.weights**2)
	def vectorize(self):
		self.v_val = np.array([part.value for part in self.particles])
	def particlelize(self):
		self.particles = [Particle(val,wei) for val,wei in itertools.izip(self.v_val,self.weights)]
	def mean(self,*args,**kwargs):
		if not self.is_vectorizable:
			raise(RuntimeError('Default ParticleSet mean method is only defined for vectorizable sets'))
		if 'axis' not in kwargs.keys():
			kwargs['axis'] = 0
		return np.mean(self.v_val,*args,**kwargs)
	def var(self,*args,**kwargs):
		if not self.is_vectorizable:
			raise(RuntimeError('Default ParticleSet var method is only defined for vectorizable sets'))
		if 'axis' not in kwargs.keys():
			kwargs['axis'] = 0
		return np.var(self.v_val,*args,**kwargs)
	def std(self,*args,**kwargs):
		if not self.is_vectorizable:
			raise(RuntimeError('Default ParticleSet std method is only defined for vectorizable sets'))
		if 'axis' not in kwargs.keys():
			kwargs['axis'] = 0
		return np.std(self.v_val,*args,**kwargs)

class TransitionProbDistribution(Distribution):
	"""
	TransitionProbDistribution class is used to sample from the
	transition probability distribution. The random number generator
	(rng) or the kernel used to sample posible particle values. rng 
	should take a particle's value as input and return another value
	sampled from the transition probability distribution. This class
	wrapps the rng's functionality and provides the sample method that
	takes a particle as input and returns another particle. The particle
	values are used in the rng call.
	"""
	def __init__(self,rng=None):
		if rng:
			self.rng = rng
		else:
			def rng(value):
				return random.gauss(value,0.1)
			self.rng = rng
	def sample(self,particle):
		return Particle(value=self.rng(particle.value),weight=particle.weight)

class VectTransitionProbDistribution(VectDistribution,TransitionProbDistribution):
	"""
	VectTransitionProbDistribution is a subclass of TransitionProbDistribution.
	It produces the same behavior but in a vectorized manner. This means
	that it is designed to return np.arrays of particle values and weights
	instead of a single Particle. The sample method is preserved but calls
	vectSample
	"""
	def __init__(self,rng=None):
		if rng:
			self.rng = rng
		else:
			def rng(values_array,*args):
				return np.random.randn(values_array.shape)*0.1+values_array
			self.rng = rng
	def vectSample(self,values_array,weights_array):
		return self.rng(values_array,weights_array)
	def sample(self,particle):
		return Particle(value=self.rng(particle.value,particle.weight),weight=particle.weight)

class ObsTransitionProbDistribution(TransitionProbDistribution):
	"""
	ObsTransitionProbDistribution class is a subclass of TransitionProbDistribution
	that is intended to sample new particles from a transition probability
	that depends not only on an initial particle's value but also on an
	observation's likelihood. For example, the metropolis hastings rule
	could be implemented as an ObsTransitionProbDistribution instance.
	The input rng is the random number generator or the kernel used to
	sample posible particle values. rng should take a particle's value
	as input and return another value sampled from the transition
	probability distribution. This class wrapps the rng's functionality
	and provides the sample method that takes an observation and a
	particle as input and returns another particle. The particle values
	are used in the rng call. The observation's loglikelihood is also
	used within the sample method.
	The class produces an instance that implements the metropolis-hastings
	rule 
	"""
	def __init__(self,loglikelihood,rng=None,iterations=1):
		"""
		Construction syntax:
		t = ObsTransitionProbDistribution(loglikelihood,rng=None,iterations=1)
		
		loglikelihood must be a function that takes two mandatory inputs
		loglikelihood(observation,value). iterations is the number of
		time to iterate the transition sampling before returning a particle
		"""
		self.loglikelihood = loglikelihood
		self.iterations = iterations
		if rng:
			self.rng = rng
		else:
			def rng(value):
				return random.gauss(value,0.1)
			self.rng = rng
	def sample(self,observation,particle):
		old_val = particle.value
		old_loglike = self.loglikelihood(observation,np.array([particle.value]))
		for it in range(self.iterations):
			new_val = self.rng(old_val)
			new_loglike = self.loglikelihood(observation,np.array([new_val]))
			# Metropolis Hastings rule
			if new_loglike>=old_loglike or random.random()<math.exp(new_loglike-old_loglike):
				old_loglike = new_loglike
				old_val = new_val
		return Particle(old_val,particle.weight*math.exp(new_loglike))

class VectObsTransitionProbDistribution(VectDistribution,ObsTransitionProbDistribution):
	"""
	VectObsTransitionProbDistribution is a subclass of ObsTransitionProbDistribution.
	It produces the same behavior but in a vectorized manner. This means
	that it is designed to return np.arrays of particle values and weights
	instead of a single Particle. The sample method is preserved but calls
	vectSample
	"""
	def __init__(self,loglikelihood,rng=None,iterations=1):
		"""
		Construction syntax:
		t = VectObsTransitionProbDistribution(loglikelihood,rng=None,iterations=1)
		
		loglikelihood must be a function that takes two mandatory inputs
		loglikelihood(observation,value). iterations is the number of
		time to iterate the transition sampling before returning a particle
		
		IMPORTANT
		loglikelihood must be able to broadcast numpy arrays as follows:
		out_values = loglikelihood(observation,in_values)
		with in_values and out_values being an np.ndarray of the same shape
		"""
		self.loglikelihood = loglikelihood
		self.iterations = iterations
		if rng:
			self.rng = rng
		else:
			def rng(values_array,*args):
				return np.random.randn(values_array.shape)*0.1+values_array
			self.rng = rng
	def vectSample(self,observation,values_array,weights_array):
		old_val = values_array
		old_loglike = self.loglikelihood(observation,old_val)
		for it in range(self.iterations):
			new_val = self.rng(old_val)
			new_loglike = self.loglikelihood(observation,new_val)
			# Metropolis Hastings rule
			update_indeces = np.logical_or(new_loglike>=old_loglike,np.random.random(new_loglike.shape)<np.exp(new_loglike-old_loglike))
			old_loglike[update_indeces] = new_loglike[update_indeces]
			old_val[update_indeces] = new_val[update_indeces]
		return old_val,weights_array*np.exp(new_loglike)
	def sample(self,observation,particle):
		return Particle(self.vectSample(observation,particle.value,particle.weight))

class PriorDistribution(Distribution):
	"""
	PriorDistribution class is used to sample from the prior
	probability distribution. The random number generator (rng) or the
	kernel used to sample posible particle values. rng should take no
	input and return a particle value sampled from the desired probability
	distribution. This class wrapps the rng's functionality and provides
	the sample method that returns a particle instead of just its value.
	"""
	def __init__(self,rng=None,weight=1.):
		if rng:
			self.rng = rng
		else:
			def rng():
				return random.gauss(0.,1.)
			self.rng = rng
	def sample(self):
		return Particle(value=self.rng(),weight=1.)

class VectPriorDistribution(VectDistribution,PriorDistribution):
	"""
	VectPriorDistribution is a subclass of PriorDistribution.
	It produces the same behavior but in a vectorized manner. This means
	that it is designed to return np.arrays of particle values and weights
	instead of a single Particle. The sample method is preserved but calls
	vectSample
	"""
	def __init__(self,rng=None,weight=1.):
		if rng:
			self.rng = rng
		else:
			def rng(n=1):
				return np.random.randn(n)
			self.rng = rng
	def vectSample(self,n):
		return self.rng(n)
	def sample(self):
		return Particle(value=self.rng(),weight=1.)

class ResamplingDistribution(Distribution):
	"""
	ResamplingDistribution class is a base class that allows to
	resample from a particle set. This class yields the same ParticleSet
	with which it was created
	"""
	def __init__(self,particle_set=None):
		if particle_set:
			self.assign(particle_set)
		self.last_sample = -1
	def assign(self,particle_set):
		self.particle_set = particle_set.copy()
		self.cumulative_probabilities = np.cumsum(np.array(self.particle_set.weights))
		self.cumulative_probabilities/= self.cumulative_probabilities[-1]
		self.cumulative_probabilities[-1] = 1.
	def sample(self):
		self.last_sample = (self.last_sample+1)%self.particle_set.N
		return self.particle_set[self.last_sample]

class VectResamplingDistribution(VectDistribution,ResamplingDistribution):
	"""
	VectResamplingDistribution is a subclass of ResamplingDistribution.
	It produces the same behavior but in a vectorized manner. This means
	that it is designed to return np.arrays of particle values and weights
	instead of a single Particle. The sample method is preserved but calls
	vectSample
	"""
	def __init__(self,particle_set=None):
		if particle_set:
			self.assign(particle_set)
		self.last_sample = -1
	def assign(self,particle_set):
		if not particle_set.is_vectorizable:
			raise(ValueError('Vectorizable resampling distributions require a vectorizable particle_set'))
		self.particle_set = particle_set.copy()
		self.cumulative_probabilities = np.cumsum(np.array(self.particle_set.weights))
		self.cumulative_probabilities/= self.cumulative_probabilities[-1]
		self.cumulative_probabilities[-1] = 1.
	def vectSample(self,n):
		indeces = np.array([(self.last_sample+1+ind)%particle_set.N for ind in range(n)])
		self.last_sample = indeces[-1]
		return self.v_val[indeces],self.weights[indeces]
	def sample(self):
		return Particle(self.vectSample(1))

class MultinomialResamplingDistribution(ResamplingDistribution):
	"""
	MultinomialResamplingDistribution class is used to sample from
	the weight distribution of particles in the selection stage. It
	takes a ParticleSet instance as input, copies it and uses its
	weights to sample particles with replacement. The sample method
	takes no input and returns a particle.
	"""
	def sample(self):
		r = random.random()
		index = np.searchsorted(self.cumulative_probabilities,r,side='left')
		self.last_sample = index
		return self.particle_set[index]

class VectMultinomialResamplingDistribution(VectResamplingDistribution):
	"""
	VectMultinomialResamplingDistribution is a subclass of MultinomialResamplingDistribution.
	It produces the same behavior but in a vectorized manner. This means
	that it is designed to return np.arrays of particle values and weights
	instead of a single Particle. The sample method is preserved but calls
	vectSample
	"""
	def vectSample(self,n):
		indeces = np.searchsorted(self.cumulative_probabilities,np.random.rand(n),side='left')
		self.last_sample = indeces[-1]
		return self.particle_set.v_val[indeces],self.particle_set.weights[indeces]
	def sample(self):
		return Particle(self.vectSample(1))

#~ class LogLikelihood(object):
	#~ """
	#~ Virtual base class Distribution that only provides a method to
	#~ sample from it
	#~ """
	#~ is_vectorizable = False
	#~ def __init__(self,evaluate):
		#~ if not callable(evaluate):
			#~ raise(TypeError('Subclasses of LogLikelihood require that the supplied evaluate method be callable'))
		#~ self.evaluate = 

class ParticleFilter:
	"""
	ParticleFilter class holds the entire sampling algorithm to
	construct the Markov Chain of particles used to sample from a
	distribution. It takes 2 obligatory inputs, the number of particles
	N and a function that returns the loglikelihood of an observation.
	The latter function should take two arguments as input, the
	observation and a particle value (that accounts for unseen causes).
	The optional inputs are the prior probability and transition
	probability distributions' rng. If input remember_markov_chain is
	set to True, it stores the particle_history variable as a list of
	ParticleSet instances.
	"""
	def __init__(self,N,loglikelihood,prior_rng=None,ptrans_rng=None,
				resample_distribution=MultinomialResamplingDistribution,
				rejuvenation_distribution=None,remember_markov_chain=False):
		self.loglikelihood = loglikelihood
		if not isinstance(prior_rng,PriorDistribution):
			self.prior = PriorDistribution(prior_rng)
		else:
			self.prior = prior_rng
		if not isinstance(ptrans_rng,TransitionProbDistribution):
			self.ptrans = TransitionProbDistribution(ptrans_rng)
		else:
			self.ptrans = ptrans_rng
		if not isinstance(resample_distribution,ResamplingDistribution):
			raise(TypeError('resample_distribution must be an instance of class ResamplingDistribution'))
		self.resample_distribution = resample_distribution
		if rejuvenation_distribution:
			if not isinstance(rejuvenation_distribution,ObsTransitionProbDistribution):
				raise(TypeError('rejuvenation_distribution must be an instance of class ObsTransitionProbDistribution'))
			self.rejuvenation_distribution = rejuvenation_distribution
		self.N = N
		self.particle_set = ParticleSet(self.N,self.prior)
		self.is_vectorizable = self.test_vectorizable()
		self.observations = []
		self.remember_markov_chain = remember_markov_chain
		if self.remember_markov_chain:
			self.particle_history = [self.particle_set.copy()]
	
	def test_vectorizable(self):
		if self.prior.is_vectorizable and self.ptrans.is_vectorizable and \
			self.resample_distribution.is_vectorizable and self.particle_set.is_vectorizable and \
			self.rejuvenation_distribution.is_vectorizable:
			return True
		else:
			return False
	
	def reset(self):
		self.particle_set = ParticleSet(self.N,self.prior)
		self.observations = []
		if self.remember_markov_chain:
			self.particle_history = [self.particle_set.copy()]
	
	# Use the transition probability on every particle to get a new ParticleSet
	def transition(self,observation):
		for index,part in enumerate(self.particle_set):
			# Get a new particle from the Markov Chain transition probability
			new_particle = self.ptrans.sample(part)
			# Update the particle's weight
			new_particle.weight = part.weight*math.exp(self.loglikelihood(observation,np.array([new_particle.value])))
			# Insert the new particle in place of the previous one
			self.particle_set.replace(index,new_particle)
	def resample(self):
		self.resample_distribution.assign(self.particle_set)
		self.particle_set = ParticleSet(N=self.particle_set.N,distribution=self.resample_distribution)
	def rejuvenate(self,observation):
		if self.rejuvenation_distribution:
			for index,part in enumerate(self.particle_set):
				# Get a new particle from the Markov Chain transition probability
				new_particle = self.rejuvenation_distribution.sample(observation,part)
				# Update the particle's weight
				new_particle.weight = part.weight*math.exp(self.loglikelihood(observation,np.array([new_particle.value])))
				# Insert the new particle in place of the previous one
				self.particle_set.replace(index,new_particle)
	def effective_sample_size(self):
		return self.particle_set.effective_sample_size()
	# A template of the particle filter algorithm
	def importance_sampling(self,observation,model=0):
		if self.is_vectorizable:
			self.vect_importance_sampling(observation,model)
			return None
		# First store the observed value
		self.observations.append(observation)
		
		self.transition(observation)
		if model>1:
			# Selection stage, where the entire particle set is resampled from the SelectionDistribution
			rtrans = self.resample_distribution.assign(self.particle_set)
			if model==2:
				self.resample()
			else:
				if self.effective_sample_size()>=0.1*self.N:
					self.resample()
				if model>3:
					self.rejuvenate(observation)
	
	def vect_importance_sampling(self,observation,model=0):
		# First store the observed value
		self.observations.append(observation)
		
		# Get a new particle from the Markov Chain transition probability
		self.particle_set.v_val = self.ptrans.vectSample(self.particle_set.v_val,self.particle_set.weights)
		# Update the particle's weight
		self.particle_set.weights*= np.exp(self.loglikelihood(observation,self.particle_set.v_val))
		if model>1:
			# Selection stage, where the entire particle set is resampled from the SelectionDistribution
			rtrans = self.resample_distribution.assign(self.particle_set)
			if model==2:
				self.particle_set.v_val,_ = self.resample_distribution.vectSample(self.N)
				self.particle_set.weights = np.ones(self.N)/self.N
			else:
				if self.effective_sample_size()>=0.1*self.N:
					self.particle_set.v_val,_ = self.resample_distribution.vectSample(self.N)
					self.particle_set.weights = np.ones(self.N)/self.N
				if model>3:
					self.particle_set.v_val,self.particle_set.weights = self.rejuvenation_distribution.vectSample(observation,self.particle_set.v_val,self.particle_set.weights)
		
		# Store a copy of the particle set if desired
		if self.remember_markov_chain:
			self.particle_set.particlelize()
			self.particle_history.append(self.particle_set.copy())

def smooth_posterior(particle_set,kernel=None,use_weights=True):
	if not kernel:
		def kernel(x,value):
			u = (np.array(x)-value)/0.1
			return (1/(math.sqrt(2*math.pi)*0.1))*np.exp(-u*u/2)
	def pkern(x):
		if use_weights:
			return np.sum(np.array([p.weight*kernel(x,p.value) for p in particle_set]),axis=0)
		else:
			return np.sum(np.array([kernel(x,p.value) for p in particle_set]),axis=0)
	return pkern

