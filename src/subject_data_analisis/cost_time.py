#!/usr/bin/python
#-*- coding: UTF-8 -*-

from __future__ import division
import numpy as np
from scipy import optimize
import math
from matplotlib import pyplot as plt

_vectErf = np.vectorize(math.erf,otypes=[np.float])
def normcdf(x,mu=0.,sigma=1.):
	"""
	Compute normal cummulative distribution with mean mu and standard
	deviation sigma. x, mu and sigma can be a numpy arrays that broadcast
	together.
	"""
	try:
		new_x = (x-mu)/sigma
	except ZeroDivisionError:
		new_x = np.sign(x-mu)*np.inf
	return 0.5 + 0.5*_vectErf(new_x / np.sqrt(2.0))

class Model():
	"""
	Class that implements the dynamic programming method that optimizes
	reward rate and computes the optimal decision bounds
	"""
	def __init__(self,model_var,prior_mu_mean=0.,prior_mu_var=1.,n=500,dt=1e-2,T=10.,\
				 reward=1.,penalty=0.,iti=1.,tp=0.,cost=0.05,store_p=True):
		"""
		Constructor input:
		model_var = True variance of the process that generates samples
		prior_mu_mean = Mean of the prior distribution on mu
		prior_mu_var = Var of the prior distribution on mu
		n = Discretization of belief space. Number of elements in g
		dt = Time steps
		T = Max time where the value is supposed to have converged already
		reward = Numerical value of reward upon success
		penalty = Numerical value of penalty upon failure
		iti = Inter trial interval
		tp = Penalty time added to iti after failure
		cost = Constant cost of accumulating new evidence
		store_p = Boolean indicating whether to store the belief transition probability array (memory expensive!)
		"""
		self.model_var = model_var
		self.prior_mu_mean = prior_mu_mean
		self.prior_mu_var = prior_mu_var
		self.n = int(n)
		if self.n%2==0:
			self.n+1
		self.g = np.linspace(0.,1.,self.n)
		self.dt = float(dt)
		self.T = float(T)
		self.t = np.arange(0.,float(T+dt),float(dt),np.float)
		self.nT = self.t.shape[0]
		self.cost = cost*np.ones_like(self.t)
		self.store_p = store_p
		self.invert_belief()
		if self.store_p:
			self.p = self.belief_transition_p()
		self.value_dp(reward=reward,penalty=penalty,iti=iti,tp=tp)
	
	def post_mu_var(self,t):
		"""
		Bayes update of the posterior variance at time t
		"""
		return 1/(t/self.model_var+1/self.prior_mu_var)
	
	def post_mu_mean(self,t,x):
		"""
		Bayes update of the posterior mean at time t with cumulated sample x
		"""
		return (x/self.model_var+self.prior_mu_mean/self.prior_mu_var)*self.post_mu_var(t)
	
	def x2g(self,t,x):
		"""
		Mapping from cumulated sample x at time t to belief
		"""
		return normcdf(self.post_mu_mean(t,x)/np.sqrt(self.post_mu_var(t)))
	
	def x2gprime(self,t,x):
		"""
		x2g(t,x) derivated by x
		"""
		return np.sqrt(0.5*self.post_mu_var(t)/np.PI)/self.model_var*\
				np.exp(-0.5*self.post_mu_mean(t,x)/np.sqrt(self.post_mu_var(t)))
	
	def x2gdot(self,t,x):
		"""
		x2g(t,x) derivated by t
		"""
		return np.sqrt(0.125*self.post_mu_var(t)**2/np.PI)/self.model_var*\
				np.exp(-0.5*self.post_mu_mean(t,x)/np.sqrt(self.post_mu_var(t)))*\
				(x/self.model_var+self.prior_mu_mean/self.prior_mu_var)
	
	def g2x(self,t,g,a=None,b=None):
		"""
		Mapping from belief at time t to cumulated sample x (inverse of x2g)
		"""
		if g==0:
			g = 1e-6
		elif g==1:
			g = 1-1e-6
		f = lambda x: self.x2g(t,x)-g
		if a is None:
			a = -100.
		if b is None:
			b = 100
		return optimize.brentq(f,a,b)
	
	def invert_belief(self):
		"""
		Invert belief vector at each time step and store in an internal array
		"""
		self.invg = np.zeros((self.nT,self.n))
		for i,t in enumerate(self.t):
			for j,g in enumerate(self.g):
				if j>0:
					self.invg[i,j] = self.g2x(t,g,self.invg[i,j-1])
				else:
					self.invg[i,j] = self.g2x(t,g)
		return self.invg
	
	def belief_transition_p(self):
		"""
		Compute the belief transition probability and store it in an internal array
		"""
		invg = self.invg
		post_var = np.zeros(self.nT)
		p = np.zeros((self.n,self.nT-1,self.n))
		for i,t in enumerate(self.t):
			post_var[i] = self.post_mu_var(t)
		for i,t in enumerate(self.t[:-1]):
			for j,g in enumerate(self.g):
				mu_n = self.post_mu_mean(t,invg[i,j])
				p[:,i,j] = -0.5*(invg[i+1]-invg[i,j]-mu_n)**2/(post_var[i]+self.model_var)+\
						   0.5*(invg[i+1]/self.model_var+self.prior_mu_mean/self.prior_mu_var)**2*post_var[i+1]
		# To avoid overflows, we substract the max value in the numerator and denominator's exponents
		p = np.exp(p-np.max(p,axis=0))/np.sum(np.exp(p-np.max(p,axis=0)),axis=0)/(self.g[1]-self.g[0])
		return np.transpose(p,(1,2,0))
	
	def value_dp(self,reward=1.,penalty=0.,iti=1.,tp=0.):
		"""
		Dynamic programming that computes the value of beliefs and the optimal bounds for decisions
		"""
		self.reward = reward
		self.penalty = penalty
		self.iti = iti
		self.tp = tp
		
		if self.store_p:
			f = lambda x: self.backpropagate_value(x)
		else:
			f = lambda x: self.memory_efficient_backpropagate_value(x)
		self.rho = optimize.brentq(f,-10,10)
		self.decision_bounds()
	
	def backpropagate_value(self,rho=None):
		if rho is not None:
			self.rho = rho
		v1 = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
		v2 = self.reward*(1-self.g)-self.penalty-(self.iti+self.g*self.tp)*self.rho
		self.value = np.zeros((self.nT,self.n))
		self.value[-1] = np.max(np.array([v1,v2]),axis=0)
		dt = self.dt
		dg = self.g[1]-self.g[0]
		for i,t in reversed(list(enumerate(self.t[:-1]))):
			v_explore = np.dot(self.p[i],self.value[i+1])*dg-(self.cost[i]+self.rho)*dt
			self.value[i] = np.max([v1,v2,v_explore],axis=0)
		return self.value[0,int(0.5*self.n)]
	
	def memory_efficient_backpropagate_value(self,rho=None):
		if rho is not None:
			self.rho = rho
		v1 = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
		v2 = self.reward*(1-self.g)-self.penalty-(self.iti+self.g*self.tp)*self.rho
		self.value = np.zeros((self.nT,self.n))
		self.value[-1] = np.max(np.array([v1,v2]),axis=0)
		dt = self.dt
		dg = self.g[1]-self.g[0]
		post_var_t1 = self.post_mu_var(self.t[-1])
		
		p = np.zeros((self.n,self.n))
		for i,t in reversed(list(enumerate(self.t[:-1]))):
			post_var_t = self.post_mu_var(t)
			for j,g in enumerate(self.g):
				mu_n = self.post_mu_mean(t,self.invg[i,j])
				p[:,j] = -0.5*(self.invg[i+1]-self.invg[i,j]-mu_n)**2/(post_var_t1+self.model_var)+\
					   0.5*post_var_t1*(self.invg[i+1]/self.model_var+self.prior_mu_mean/self.prior_mu_var)**2
			# We transpose the array after the computation so numpy can correctly broadcast the intermediate operations (sum and max)
			p = np.transpose(np.exp(p-np.max(p,axis=0))/np.sum(np.exp(p-np.max(p,axis=0)),axis=0)/dg,(1,0))
			v_explore = np.dot(p,self.value[i+1])*dg-(self.cost[i]+self.rho)*dt
			self.value[i] = np.max([v1,v2,v_explore],axis=0)
			post_var_t1 = post_var_t
		return self.value[0,int(0.5*self.n)]
	
	def decision_bounds(self):
		self.bounds = np.zeros((self.nT,2))
		v1 = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
		v2 = self.reward*(1-self.g)-self.penalty-(self.iti+self.g*self.tp)*self.rho
		decide_1 = np.abs((v1-self.value)/v1)<1e-9
		decide_2 = np.abs((v2-self.value)/v1)<1e-9
		for i,t in enumerate(self.t):
			if any(decide_1[i]):
				bound1 = self.g[decide_1[i].nonzero()[0][0]]
			else:
				bound1 = 1.
			if any(decide_2[i]):
				bound2 = self.g[decide_2[i].nonzero()[0][-1]]
			else:
				bound2 = 0.
			self.bounds[i] = np.array([bound1,bound2])
		return self.bounds
	
	def refine_value(self,dt=None,n=None):
		change = False
		if dt is not None:
			if dt<self.dt:
				oldt = self.t.copy()
				self.t = np.arange(0,self.T+dt,dt,np.float)
				self.nT = self.t.shape[0]
				self.cost = np.interp(self.t, oldt, self.cost)
				change = True
		if n is not None:
			n = int(n)
			if n%2==0:
				n+=1
			if n>self.n:
				self.n = n
				self.g = np.linspace(0.,1.,self.n)
				change = True
		if change:
			self.invert_belief()
			if self.store_p:
				self.p = self.belief_transition_p()
				self.backpropagate_value(rho=self.rho)
			else:
				self.memory_efficient_backpropagate_value(rho=self.rho)
			self.decision_bounds()

def test():
	m = Model(model_var=10,prior_mu_var=40,n=51,T=10.,cost=5,store_p=False)
	m.refine_value(n=501)
	b = m.bounds
	plt.figure(figsize=(13,10))
	plt.subplot(121)
	plt.imshow(m.value.T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[m.t[0],m.t[-1],m.g[0],m.g[-1]])
	cbar = plt.colorbar(orientation='horizontal')
	cbar.set_label('Value')
	plt.ylabel('belief')
	plt.xlabel('T [s]')
	#~ colors = plt.get_cmap('jet')
	#~ plt.gca().set_color_cycle([colors(v) for v in np.linspace(0,1,m.value.shape[0])])
	#~ plt.plot(m.g,m.value.T)
	#~ plt.xlabel('belief')
	#~ plt.ylabel('value')
	plt.subplot(122)
	plt.plot(m.t,b)
	plt.xlabel('T [s]')
	plt.ylabel('Bound')
	plt.show()

if __name__=="__main__":
	test()
