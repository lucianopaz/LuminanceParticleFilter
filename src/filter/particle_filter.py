#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package defining a basic particle filter """

from __future__ import division
import math, numpy, random, scipy, copy

class Particle:
	""" Particle class has two properties, its weight (the log of the
	weight is also stored) and its value. The value is the random
	variable obtained when sampling from a Markov Chain """
	def __init__(self,value=None,weight=0):
		self.value = numpy.array(value)
		self.weight = weight
		try:
			logweight = math.log(weight)
		except ValueError:
			logweight = -float("inf")
		self.logweight = logweight
	def copy(self):
		return Particle(copy.copy(self.value),self.weight)

class Distribution:
	""" Virtual base class Distribution that only provides a method to
	sample from it """
	def sample(self):
		return Particle()

class ParticleSet:
	""" ParticleSet class works as a special particle container that
	stores the number of particles, the particle objects and a list
	of weights for faster access """
	def __init__(self,N=1,distribution=Distribution()):
		self.N = int(N)
		self.weights = numpy.array([1./self.N for _ in range(N)])
		self.particles = [distribution.sample() for _ in range(N)]
		for i,part in enumerate(self.particles):
			part.weight = self.weights[i]
			part.logweight = math.log(self.weights[i])
		self.vectorize()
	def append(self,part):
		self.N+=1
		self.weights.append(part.weight)
		self.particles.append(part)
		self.vectorize()
	def pop(self,part,index=-1):
		self.N-=1
		self.weigths.pop(index)
		ret =  self.particles.pop(index)
		self.vectorize()
		return ret
	def copy(self):
		copied_set = ParticleSet(N=0)
		copied_set.N = self.N
		copied_set.weights = copy.copy(self.weights)
		copied_set.particles = [part.copy() for part in self.particles]
		copied_set.vectorize()
		return copied_set
	def replace(self,index,particle):
		self.particles[index] = particle
		self.weights[index] = particle.weight
		self.v_val[index] = particle.value
	def __getitem__(self,index):
		if index>=self.N or index<-self.N:
			raise(IndexError)
		return self.particles[index]
	def effective_sample_size(self):
		# Kish approximate sample size formula
		return numpy.sum(self.weights)**2/numpy.sum(self.weights**2)
	def vectorize(self):
		self.v_val = numpy.array([part.value for part in self.particles])
	def mean(self,*args,**kwargs):
		if 'axis' not in kwargs.keys():
			kwargs['axis'] = 0
		return numpy.mean(self.v_val,*args,**kwargs)
	def var(self,*args,**kwargs):
		if 'axis' not in kwargs.keys():
			kwargs['axis'] = 0
		return numpy.var(self.v_val,*args,**kwargs)
	def std(self,*args,**kwargs):
		if 'axis' not in kwargs.keys():
			kwargs['axis'] = 0
		return numpy.std(self.v_val,*args,**kwargs)

class TransitionProbDistribution(Distribution):
	""" TransitionProbDistribution class is used to sample from the
	transition probability distribution. The random number generator
	(rng) or the kernel used to sample posible particle values. rng 
	should take a particle's value as input and return another value
	sampled from the transition probability distribution. This class
	wrapps the rng's functionality and provides the sample method that
	takes a particle as input and returns another particle. The particle
	values are used in the rng call. """
	def __init__(self,rng=None):
		if rng:
			self.rng = rng
		else:
			def rng(value):
				return random.gauss(value,0.1)
			self.rng = rng
	def sample(self,particle):
		return Particle(value=self.rng(particle.value),weight=particle.weight)

class PriorDistribution(Distribution):
	""" PriorDistribution class is used to sample from the prior
	probability distribution. The random number generator (rng) or the
	kernel used to sample posible particle values. rng should take no
	input and return a particle value sampled from the desired probability
	distribution. This class wrapps the rng's functionality and provides
	the sample method that returns a particle instead of just its value. """
	def __init__(self,rng=None,weight=1.):
		if rng:
			self.rng = rng
		else:
			def rng():
				return random.gauss(0.,1.)
			self.rng = rng
	def sample(self):
		return Particle(value=self.rng(),weight=1.)

class ResamplingDistribution(Distribution):
	""" ResamplingDistribution class is a base class that allows to
	resample from a particle set. This class yields the same particle_set
	with which it was created """
	def __init__(self,particle_set):
		self.assign(particle_set)
		self.last_sample = -1
	def assign(self,particle_set):
		self.particle_set = particle_set.copy()
		self.cumulative_probabilities = numpy.cumsum(numpy.array(self.particle_set.weights))
		self.cumulative_probabilities/= self.cumulative_probabilities[-1]
		self.cumulative_probabilities[-1] = 1.
	def sample(self):
		self.last_sample = (self.last_sample+1)%self.particle_set.N
		return self.particle_set[self.last_sample]

class MultinomialResamplingDistribution(ResamplingDistribution):
	""" MultinomialResamplingDistribution class is used to sample from
	the weight distribution of particles in the selection stage. It
	takes a ParticleSet instance as input, copies it and uses its
	weights to sample particles with replacement. The sample method
	takes no input and returns a particle. """
	def sample(self):
		r = random.random()
		index = numpy.searchsorted(self.cumulative_probabilities,r,side='left')
		self.last_sample = index
		return self.particle_set.particles[index]

class ParticleFilter:
	""" ParticleFilter class holds the entire sampling algorithm to
	construct the Markov Chain of particles used to sample from a
	distribution. It takes 2 obligatory inputs, the number of particles
	N and a function that returns the loglikelihood of an observation.
	The latter function should take two arguments as input, the
	observation and a particle value (that accounts for unseen causes).
	The optional inputs are the prior probability and transition
	probability distributions' rng. If input remember_markov_chain is
	set to True, it stores the particle_history variable as a list of
	ParticleSet instances. """
	def __init__(self,N,loglikelihood,prior_rng=None,ptrans_rng=None,resample_distribution=MultinomialResamplingDistribution,remember_markov_chain=False):
		self.loglikelihood = loglikelihood
		if not isinstance(prior_rng,PriorDistribution):
			self.prior = PriorDistribution(prior_rng)
		else:
			self.prior = prior_rng
		if not isinstance(ptrans_rng,TransitionProbDistribution):
			self.ptrans = TransitionProbDistribution(ptrans_rng)
		else:
			self.ptrans = ptrans_rng
		#~ if not isinstance(resample_distribution,ResamplingDistribution):
			#~ raise(TypeError)
		self.resample_distribution = resample_distribution
		self.N = N
		self.particle_set = ParticleSet(self.N,self.prior)
		self.observations = None
		self.remember_markov_chain = remember_markov_chain
		if self.remember_markov_chain:
			self.particle_history = [self.particle_set.copy()]
	
	def reset(self):
		self.particle_set = ParticleSet(self.N,self.prior)
		self.observations = None
		if self.remember_markov_chain:
			self.particle_history = [self.particle_set.copy()]
	
	# Use the transition probability on every particle to get a new ParticleSet
	def transition(self,observation):
		for index,part in enumerate(self.particle_set):
			# Get a new particle from the Markov Chain transition probability
			new_particle = self.ptrans.sample(part)
			# Update the particle's weight
			new_particle.logweight = part.logweight+self.loglikelihood(observation,new_particle.value)
			new_particle.weight = math.exp(new_particle.logweight)
			# Insert the new particle in place of the previous one
			self.particle_set.replace(index,new_particle)
	def resample(self):
		self.particle_set = ParticleSet(N=self.particle_set.N,distribution=self.resample_distribution(self.particle_set))
	def metropolis_hastings(self,observation,iterations=10,ptrans=None):
		if not ptrans:
			ptrans = self.ptrans
		for index,orig_part in enumerate(self.particle_set):
			part = orig_part.copy()
			old_loglike = self.loglikelihood(observation,part.value)
			for it in range(iterations):
				new_particle = ptrans.sample(part)
				new_loglike = self.loglikelihood(observation,new_particle.value)
				# Metropolis Hastings rule
				if new_loglike>=old_loglike or random.random()<math.exp(new_loglike-old_loglike):
					old_loglike = new_loglike
					part = new_particle
			part.logweight = part.logweight+old_loglike
			part.weight = math.exp(part.logweight)
			self.particle_set.replace(index,part)
	def effective_sample_size(self):
		return self.particle_set.effective_sample_size()
	# A template of the particle filter algorithm
	def importance_sampling(self,observation):
		# First store the observed value
		self.observations.append(observation)
		# Importance sampling stage
		self.transition(observation)
		# Selection stage, where the entire particle set is resampled from the SelectionDistribution
		self.resample()
		# Store a copy of the particle set if desired
		if self.remember_markov_chain:
			self.particle_history.append(self.particle_set.copy())

def smooth_posterior(particle_set,kernel=None,use_weights=True):
	if not kernel:
		def kernel(x,value):
			u = (numpy.array(x)-value)/0.1
			return (1/(math.sqrt(2*math.pi)*0.1))*numpy.exp(-u*u/2)
	def pkern(x):
		if use_weights:
			return numpy.sum(numpy.array([p.weight*kernel(x,p.value) for p in particle_set]),axis=0)
		else:
			return numpy.sum(numpy.array([kernel(x,p.value) for p in particle_set]),axis=0)
	return pkern

