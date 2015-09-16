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
	deviation sigma. x can be a numpy array.
	"""
	try:
		new_x = (x-mu)/sigma
	except ZeroDivisionError:
		new_x = np.sign(x-mu)*np.inf
	return 0.5 + 0.5*_vectErf(new_x / np.sqrt(2.0))

class Model():
	def __init__(self,model_var,prior_mu_mean=0.,prior_mu_var=1.,n=500,dt=1e-2,T=10.,\
				 reward=1.,penalty=0.,iti=1.,tp=0.,cost=0.05):
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
		self.p = self.belief_transition_p()
		self.cost = cost*np.ones_like(self.t)
		self.value_dp(reward=reward,penalty=penalty,iti=iti,tp=tp)
	
	def post_mu_var(self,t):
		return 1/(t/self.model_var+1/self.prior_mu_var)
	
	def post_mu_mean(self,t,x):
		return (x/self.model_var+self.prior_mu_mean/self.prior_mu_var)*self.post_mu_var(t)
	
	def x2g(self,t,x):
		return normcdf(self.post_mu_mean(t,x)/np.sqrt(self.post_mu_var(t)))
	
	def x2gprime(self,t,x):
		return np.exp(-0.5*self.post_mu_mean(t,x)/np.sqrt(self.post_mu_var(t)))
	
	def g2x(self,t,g,a=None,b=None):
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
	
	def belief_transition_p(self):
		invg = np.zeros((self.nT,self.n))
		post_var = np.zeros(self.nT)
		p = np.zeros((self.n,self.nT-1,self.n))
		
		for i,t in enumerate(self.t):
			for j,g in enumerate(self.g):
				if j>0:
					invg[i,j] = self.g2x(t,g,invg[i,j-1])
				else:
					invg[i,j] = self.g2x(t,g)
			post_var[i] = self.post_mu_var(t)
		for i,t in enumerate(self.t[:-1]):
			for j,g in enumerate(self.g):
				mu_n = self.post_mu_mean(t,invg[i,j])
				for k,g_1 in enumerate(self.g):
					p[k,i,j] = -0.5*(invg[i+1,k]-invg[i,j]-mu_n)**2/(post_var[i]+self.model_var)+\
							   0.5*(invg[i+1,k]/self.model_var+self.prior_mu_mean/self.prior_mu_var)**2*post_var[i+1]
		# To avoid overflows, we substract the max value in the numerator and denominator's exponents
		p = np.exp(p-np.max(p,axis=0))/np.sum(np.exp(p-np.max(p,axis=0)),axis=0)/(self.g[1]-self.g[0])
		return np.transpose(p,(1,2,0))
	
	def value_dp(self,reward=1.,penalty=0.,iti=1.,tp=0.):
		self.reward = reward
		self.penalty = penalty
		self.iti = iti
		self.tp = tp
		
		f = lambda x: self.backpropagate_value(x)
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
			v_explore = np.sum(self.value[i+1]*self.p[i]*dg,axis=1)-(self.cost[i]+self.rho)*dt
			self.value[i] = np.max([v1,v2,v_explore],axis=0)
		output = self.value[0,int(0.5*self.n)]
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
	
	def refine_value(self,dt,n):
		change = False
		if dt<self.dt:
			oldt = self.t.copy()
			self.t = np.arange(0,self.T+dt,dt,np.float)
			self.nT = self.t.shape[0]
			self.cost = np.interp(self.t, oldt, self.cost)
			change = True
		n = int(n)
		if n%2==0:
			n+=1
		if n>self.n:
			self.n = n
			self.g = np.linspace(0.,1.,self.n)
			change = True
		if change:
			self.p = self.belief_transition_p()
			self.backpropagate_value(rho=self.rho)
			self.decision_bounds()

def test():
	m = Model(model_var=10,prior_mu_var=40,n=101,T=10.,cost=1)
	#~ print m.rho
	#~ m.refine_value(dt=1e-2,n=501)
	b = m.bounds
	plt.figure(figsize=(13,10))
	colors = plt.get_cmap('jet')
	plt.subplot(121)
	plt.gca().set_color_cycle([colors(v) for v in np.linspace(0,1,m.value.shape[0])])
	plt.plot(m.g,m.value.T)
	plt.xlabel('belief')
	plt.ylabel('value')
	plt.subplot(122)
	plt.plot(m.t,b)
	plt.xlabel('T [s]')
	plt.ylabel('Bound')
	plt.show()

if __name__=="__main__":
	test()
