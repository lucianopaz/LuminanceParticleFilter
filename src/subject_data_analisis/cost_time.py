#!/usr/bin/python
#-*- coding: UTF-8 -*-

from __future__ import division
import numpy as np
from scipy import optimize

class Model():
	def __init__(self,prior,n,dt,nT,m,mu_range,reward=1.,penalty=0.,T_i=1.,T_p=0.):
		self.n = int(n)
		self.dt = float(dt)
		self.nT = int(nT)
		self.reward = float(reward)
		self.penalty = float(penalty)
		self.T_i = float(T_i)
		self.T_p = float(T_p)
		self.m = int(m) # Discretization of prior mu values
		self.mu = np.linspace(mu_range[0],mu_range[1],self.m)
		self.g = np.linspace(0,1,n)
		self.V = np.zeros((nT,n))
		self.costs = np.zeros(nT)
		if isinstance(prior,np.ndarray):
			if prior.shape~=self.mu.shape:
				raise ValueError("Supplied prior must have shape (%d)"%(self.m))
			self.prior_mu = prior
		elif callable(prior):
			self.prior_mu = np.array([prior(mu) for mu in self.mu])
		else:
			raise TypeError("Supplied prior must be an ndarray or a function to evaluate prior mu probabilities")
		self.p = self.propagate_probability()
	
	def propagate_probability(self):
		p = np.zeros((nT,n,n))
	
	def g2x(self,g,p=self.prior_mu):
		f = lambda x: self.x2g(x,p)-g
		return optimize.brentq(f,self.mu[0]-10*(self.mu[-1]-self.mu[0]),self.mu[-1]+10*(self.mu[-1]-self.mu[0]),\
							   args=(p,))
	
	def x2g(self,x,p=self.prior_mu):
		arg = self.mu*(x-0.5*self.mu)
		max_arg = np.max(arg)
		norm_prob = p*np.exp(arg/max_arg)
		return np.sum(norm_prob[self.mu>=0])/np.sum(norm_prob)

def value_backtrack(reward_rate,V,T_i,T_p,g,reward_1,reward_2,p):
	

def test():
	n = 501                         # Discretization of belief values
	maxRT = 5.                      # Maximum response time
	T = maxRT*10.                   # Time assumed to be long enough to fix the value of the beliefs
	dt = 0.04                       # Time step (time between each brightness refresh)
	nT = int(np.ceil(T/dt))         # Number of time steps
	reward = 1.                     # Reward for correct decision (we assume a symmetric reward)
	penalty = 0.                    # Penalty for misses
	T_i = 1.                        # Inter trial time
	T_p = 0.                        # Time penalty after wrong answer
	g = np.linspace(0,1,n)          # belief of choosing option 1. (1-g) belief of choosing option 2
	V = np.ones((nT,n))             # Value of belief for each time
	costs = np.zeros(nT)            # Cost of gathering more evidence at each time step
	p = np.zeros((nT,n,n))          # Belief transition density
	
	reward_rate_interval = np.array([penalty,reward])
	reward_rate = np.mean(reward_rate_interval) # Espected reward rate that is iteratively adjusted to get V[0][int(0.5*n)]==0
	
	# Set the final value for the beliefs
	tolV = 1e-3
	max_steps = 1e3
	steps = 0
	reward_1 = g*reward + (1-g)*penalty
	reward_2 = (1-g)*reward + g*penalty
	
	brentq
	while abs(V[0][int(0.5*n)])>=tolV or steps<max_steps:
		if max_steps>0:
			if V[0][int(0.5*n)]>0:
				reward_rate_interval[1] = np.mean(reward_rate_interval)
			else:
				reward_rate_interval[0] = np.mean(reward_rate_interval)
			reward_rate = np.mean(reward_rate_interval)
		value_of_selecting_1 = reward_1-(T_i+(1-g)*T_p)*reward_rate
		value_of_selecting_2 = reward_2-(T_i+g*T_p)*reward_rate
		V[-1] = np.max(np.array([value_of_selecting_1,value_of_selecting_2]),axis=0)
		for t in range(nT-1).reverse():
			value_of_exploring = 
			V[t] = np.max(np.array([value_of_selecting_1,value_of_selecting_2,]),axis=0)

if __name__=="__main__":
	test()
