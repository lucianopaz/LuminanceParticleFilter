#!/usr/bin/python
#-*- coding: UTF-8 -*-

from __future__ import division
import numpy as np
from scipy import optimize
import math

_vectErf = np.vectorize(math.erf,otypes=[np.float])
def normcdf(x,mu=0.,sigma=1.):
	"""
	Compute normal cummulative distribution with mean mu and standard
	deviation sigma. x can be a numpy array.
	"""
	try:
		new_x = (x-mu)/sigma
	except ZeroDivisionError:
		new_x = np.sign(x-mu)*np.inf
	return 0.5 + 0.5*_vectErf(new_x / np.sqrt(2.0))

#~ class Model():
	#~ def __init__(self,prior,n=1,dt=0.1,nT=1,m=1,mu_range,reward=1.,penalty=0.,T_i=1.,T_p=0.):
		#~ self.n = int(n)
		#~ self.dt = float(dt)
		#~ self.nT = int(nT)
		#~ self.reward = float(reward)
		#~ self.penalty = float(penalty)
		#~ self.T_i = float(T_i)
		#~ self.T_p = float(T_p)
		#~ self.m = int(m) # Discretization of prior mu values
		#~ self.mu = np.linspace(mu_range[0],mu_range[1],self.m)
		#~ self.g = np.linspace(0,1,n)
		#~ self.V = np.zeros((nT,n))
		#~ self.costs = np.zeros(nT)
		#~ if isinstance(prior,np.ndarray):
			#~ if prior.shape!=self.mu.shape:
				#~ raise ValueError("Supplied prior must have shape (%d)"%(self.m))
			#~ self.prior_mu = prior
		#~ elif callable(prior):
			#~ self.prior_mu = np.array([prior(mu) for mu in self.mu])
		#~ else:
			#~ raise TypeError("Supplied prior must be an ndarray or a function to evaluate prior mu probabilities")
		#~ self.p = self.propagate_probability()
	
	#~ def propagate_probability(self):
		#~ p = np.zeros((nT,n,n))
	
class Model():
	def __init__(self,model_var,prior_mu_mean=0.,prior_mu_var=1.,n=500,dt=1e-2,T=10.):
		self.model_var = model_var
		self.prior_mu_mean = prior_mu_mean
		self.prior_mu_var = prior_mu_var
		self.n = int(n)
		self.g = np.linspace(0,1,self.n)
		self.t = np.arange(0,T+dt,dt,np.float)
		self.nT = self.t.shape[0]-1
	
	def post_mu_var(self,t):
		return 1/(t/self.model_var+1/self.prior_mu_var)
	
	def post_mu_mean(self,t,x):
		return (x/self.model_var+self.prior_mu_mean/self.prior_mu_var)*self.post_mu_var(t)
	
	def x2g(self,t,x):
		return normcdf(self.post_mu_mean(t,x)/np.sqrt(self.post_mu_var(t)))
	
	def x2gprime(self,t,x):
		return np.exp(-0.5*self.post_mu_mean(t,x)/np.sqrt(self.post_mu_var(t)))
	
	def g2x(self,t,g,a=None,b=None):
		f = lambda x: self.x2g(t,x)-g
		if a is None:
			a = -100.
		if b is None:
			b = 100
		return optimize.brentq(f,a,b)
	
	def belief_transition_p(self):
		xx = np.zeros(self.nT,self.n)
		for i,t in enumerate(self.t[:-1]):
			for j,g in enumerate(self.g)
				xx[i,j] = self.g2x(t,g)
				
		

#~ def value_backtrack(reward_rate,V,T_i,T_p,g,reward_1,reward_2,p):
	#~ 

#~ def test():
	#~ n = 501                         # Discretization of belief values
	#~ maxRT = 5.                      # Maximum response time
	#~ T = maxRT*10.                   # Time assumed to be long enough to fix the value of the beliefs
	#~ dt = 0.04                       # Time step (time between each brightness refresh)
	#~ nT = int(np.ceil(T/dt))         # Number of time steps
	#~ reward = 1.                     # Reward for correct decision (we assume a symmetric reward)
	#~ penalty = 0.                    # Penalty for misses
	#~ T_i = 1.                        # Inter trial time
	#~ T_p = 0.                        # Time penalty after wrong answer
	#~ g = np.linspace(0,1,n)          # belief of choosing option 1. (1-g) belief of choosing option 2
	#~ V = np.ones((nT,n))             # Value of belief for each time
	#~ costs = np.zeros(nT)            # Cost of gathering more evidence at each time step
	#~ p = np.zeros((nT,n,n))          # Belief transition density
	#~ 
	#~ reward_rate_interval = np.array([penalty,reward])
	#~ reward_rate = np.mean(reward_rate_interval) # Espected reward rate that is iteratively adjusted to get V[0][int(0.5*n)]==0
	#~ 
	#~ # Set the final value for the beliefs
	#~ tolV = 1e-3
	#~ max_steps = 1e3
	#~ steps = 0
	#~ reward_1 = g*reward + (1-g)*penalty
	#~ reward_2 = (1-g)*reward + g*penalty
	#~ 
	#~ brentq
	#~ while abs(V[0][int(0.5*n)])>=tolV or steps<max_steps:
		#~ if max_steps>0:
			#~ if V[0][int(0.5*n)]>0:
				#~ reward_rate_interval[1] = np.mean(reward_rate_interval)
			#~ else:
				#~ reward_rate_interval[0] = np.mean(reward_rate_interval)
			#~ reward_rate = np.mean(reward_rate_interval)
		#~ value_of_selecting_1 = reward_1-(T_i+(1-g)*T_p)*reward_rate
		#~ value_of_selecting_2 = reward_2-(T_i+g*T_p)*reward_rate
		#~ V[-1] = np.max(np.array([value_of_selecting_1,value_of_selecting_2]),axis=0)
		#~ for t in range(nT-1).reverse():
			#~ value_of_exploring = 
			#~ V[t] = np.max(np.array([value_of_selecting_1,value_of_selecting_2,]),axis=0)


def belief_transition_logp():
	x_prior = g2x(g)
	var_t = post_mu_var(t)
	mu_t = post_mu_mean(t)
	var_t1 = post_mu_var(t+dt)
	return -0.5((x-x_prior_mu_t)**2/(var_t+model_var)+(x/model_var+prior_mu_mean/prior_mu_var)**2*var_t1)

if __name__=="__main__":
	test()
