#!/usr/bin/python
#-*- coding: UTF-8 -*-

from __future__ import division
import numpy as np
from scipy import io
from scipy import optimize
import math, copy
from matplotlib import pyplot as plt
from utils import normcdf,normcdfinv

class DecisionPolicy():
	"""
	Class that implements the dynamic programming method that optimizes
	reward rate and computes the optimal decision bounds
	"""
	def __init__(self,model_var,prior_mu_mean=0.,prior_mu_var=1.,n=500,dt=1e-2,T=10.,\
				 reward=1.,penalty=0.,iti=1.,tp=0.,cost=0.05,store_p=True,\
				 compute_value=True):
		"""
		Constructor input:
		model_var = True variance of the process that generates samples per unit time. This value is then multiplied by dt.
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
		self.model_var = model_var*dt
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
		self.reward = reward
		self.penalty = penalty
		self.iti = iti
		self.tp = tp
		
		if compute_value:
			self.invert_belief()
			if self.store_p:
				self.p = self.belief_transition_p()
			self.value_dp()
	
	def reset(self):
		if self.store_p:
			self.p = np.empty(1)
		self.invg = np.empty(1)
		self.value = np.empty(1)
		self.bounds = np.empty(1)
	
	def copy(self):
		out = DecisionPolicy(self.model_var,self.prior_mu_mean,self.prior_mu_var,self.n,self.dt,self.T,\
				 self.reward,self.penalty,self.iti,self.tp,cost=0.05,store_p=self.store_p,compute_value=False)
		out.invg = copy.deepcopy(self.invg)
		if self.store_p:
			out.p = copy.deepcopy(self.p)
		out.value = copy.deepcopy(self.value)
		out.bounds = copy.deepcopy(self.bounds)
		out.cost = copy.deepcoy(self.cost)
		return out
	
	def set_constant_cost(self,cost):
		self.cost = cost*np.ones_like(self.cost)
	
	def post_mu_var(self,t):
		"""
		Bayes update of the posterior variance at time t
		"""
		return 1./(t/self.model_var+1./self.prior_mu_var)
	
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
	
	def g2x(self,t,g):
		"""
		Mapping from belief at time t to cumulated sample x (inverse of x2g)
		"""
		return self.model_var*(normcdfinv(g)/np.sqrt(self.post_mu_var(t))-self.prior_mu_mean/self.prior_mu_var)
	
	def invert_belief(self):
		"""
		Invert belief vector at each time step and store in an internal array
		"""
		self.invg = np.zeros((self.nT,self.n))
		for i,t in enumerate(self.t):
			for j,g in enumerate(self.g):
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
				#~ p[:,i,j] = 0.5*normcdfinv(self.g)**2*post_var_t[i+1]/post_var_t**2-\
						   #~ 0.5*self.model_var/(self.model_var+post_var[i])*(normcdfinv(self.g)/post_var[i+1]-normcdfinv(g)/post_var[i]-mu_n)**2
		# To avoid overflows, we substract the max value in the numerator and denominator's exponents
		p = np.exp(p-np.max(p,axis=0))/np.sum(np.exp(p-np.max(p,axis=0)),axis=0)#/(self.g[1]-self.g[0])
		return np.transpose(p,(1,2,0))
	
	def test_belief_transition_p(self):
		"""
		Compute the belief transition probability and store it in an internal array
		"""
		invg = self.invg
		post_var = np.zeros(self.nT)
		p = np.zeros((self.n*(self.nT-1),self.n))
		for i,t in enumerate(self.t):
			post_var[i] = self.post_mu_var(t)
		counter = -1
		for i,t in reversed(list(enumerate(self.t[:-1]))):
			counter+=1
			for j,g in enumerate(self.g):
				mu_n = self.post_mu_mean(t,invg[i,j])
				p[counter*self.n+j] = -0.5*(invg[i+1]-invg[i,j]-mu_n)**2/(post_var[i]+self.model_var)+\
						         0.5*(invg[i+1]/self.model_var+self.prior_mu_mean/self.prior_mu_var)**2*post_var[i+1]
				#~ p[:,i,j] = 0.5*normcdfinv(self.g)**2*post_var_t[i+1]/post_var_t**2-\
						   #~ 0.5*self.model_var/(self.model_var+post_var[i])*(normcdfinv(self.g)/post_var[i+1]-normcdfinv(g)/post_var[i]-mu_n)**2
		# To avoid overflows, we substract the max value in the numerator and denominator's exponents
		#~ p = np.exp(p-np.max(p,axis=1))/np.sum(np.exp(p-np.max(p,axis=1)),axis=1)#/(self.g[1]-self.g[0])
		return p #np.transpose(p,(1,2,0))
	
	def value_dp(self,lb=-10.,ub=10.):
		"""
		Method that call the dynamic programming method that computes
		the value of beliefs and the optimal bounds for decisions,
		adjusting the predicted average reward (rho)
		"""
		if self.store_p:
			f = lambda x: self.backpropagate_value(x)
		else:
			f = lambda x: self.memory_efficient_backpropagate_value(x)
		try:
			self.rho = optimize.brentq(f,lb,ub)
		except ValueError as er:
			if er.message=="f(a) and f(b) must have different signs":
				#~ print "Changing bound for brentq root finding of rho"
				m = f(lb)
				M = f(ub)
				if m!=0 and M!=0:
					while np.sign(m)==np.sign(M):
						if (m<M and m<0) or (m>M and m>0):
							lb = ub
							ub = ub*10
							#~ print "Setting upper bound equal to %1.4f" % (ub)
							M = f(ub)
						elif (m>M and m<0) or (m<M and m>0):
							ub = lb
							lb = lb*10
							#~ print "Setting lower bound equal to %1.4f" % (lb)
							m = f(lb)
					#~ print "Success! Bounds for root finding set to lb=%1.4f and ub=%1.4f" % (lb,ub)
					self.rho = optimize.brentq(f,lb,ub)
				else:
					if m==0:
						self.rho = lb
					else:
						self.rho = ub
			else:
				raise er
		self.decision_bounds()
	
	def backpropagate_value(self,rho=None):
		"""
		Dynamic programming method that computes the value of beliefs.
		It uses the previously computed belief transition probability
		density that is very expensive in terms of memory.
		"""
		if rho is not None:
			self.rho = rho
		v1 = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
		v2 = self.reward*(1-self.g)-self.penalty*self.g-(self.iti+self.g*self.tp)*self.rho
		self.value = np.zeros((self.nT,self.n))
		self.value[-1] = np.max(np.array([v1,v2]),axis=0)
		dt = self.dt
		#~ dg = self.g[1]-self.g[0]
		for i,t in reversed(list(enumerate(self.t[:-1]))):
			#~ v_explore = np.dot(self.p[i],self.value[i+1])*dg-(self.cost[i]+self.rho)*dt
			v_explore = np.dot(self.p[i],self.value[i+1])-(self.cost[i]+self.rho)*dt
			self.value[i] = np.max([v1,v2,v_explore],axis=0)
		return self.value[0,int(0.5*self.n)]
	
	def memory_efficient_backpropagate_value(self,rho=None):
		"""
		Dynamic programming method that computes the value of beliefs.
		It computes the belief transition probability density on the fly
		and is thus memory efficient at the expense of execution time.
		"""
		if rho is not None:
			self.rho = rho
		v1 = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
		v2 = self.reward*(1-self.g)-self.penalty*self.g-(self.iti+self.g*self.tp)*self.rho
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
				p[:,j] = -0.5*(self.invg[i+1]-self.invg[i,j]-mu_n)**2/(post_var_t+self.model_var)+\
					   0.5*post_var_t1*(self.invg[i+1]/self.model_var+self.prior_mu_mean/self.prior_mu_var)**2
				#~ p[:,j] = 0.5*normcdfinv(self.g)**2*post_var_t1/post_var_t**2-\
						 #~ 0.5*self.model_var/(self.model_var+post_var_t)*(normcdfinv(self.g)/post_var_t1-normcdfinv(g)/post_var_t-mu_n)**2
			# We transpose the array after the computation so numpy can correctly broadcast the intermediate operations (sum and max)
			p = np.transpose(np.exp(p-np.max(p,axis=0))/np.sum(np.exp(p-np.max(p,axis=0)),axis=0),(1,0))
			v_explore = np.dot(p,self.value[i+1])-(self.cost[i]+self.rho)*dt
			#~ v_explore = np.dot(p,self.value[i+1])*dg-(self.cost[i]+self.rho)*dt
			self.value[i] = np.max([v1,v2,v_explore],axis=0)
			post_var_t1 = post_var_t
		return self.value[0,int(0.5*self.n)]
	
	def v_explore(self,rho=None):
		if rho is not None:
			rho = self.rho
		v1 = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
		v2 = self.reward*(1-self.g)-self.penalty*self.g-(self.iti+self.g*self.tp)*self.rho
		v_explore = np.zeros((self.nT-1,self.n))
		dt = self.dt
		post_var_t1 = self.post_mu_var(self.t[-1])
		
		p = np.zeros((self.n,self.n))
		for i,t in reversed(list(enumerate(self.t[:-1]))):
			post_var_t = self.post_mu_var(t)
			for j,g in enumerate(self.g):
				mu_n = self.post_mu_mean(t,self.invg[i,j])
				p[:,j] = -0.5*(self.invg[i+1]-self.invg[i,j]-mu_n)**2/(post_var_t1+self.model_var)+\
					   0.5*post_var_t1*(self.invg[i+1]/self.model_var+self.prior_mu_mean/self.prior_mu_var)**2
			# We transpose the array after the computation so numpy can correctly broadcast the intermediate operations (sum and max)
			p = np.transpose(np.exp(p-np.max(p,axis=0))/np.sum(np.exp(p-np.max(p,axis=0)),axis=0),(1,0))
			v_explore[i] = np.dot(p,self.value[i+1])-(self.cost[i]+self.rho)*dt
			post_var_t1 = post_var_t
		return v_explore
	
	def decision_bounds(self,value=None):
		"""
		Compute the decision bounds from the value of the beliefs
		"""
		if value is None:
			value = self.value
		self.bounds = np.zeros((2,self.nT))
		v1_arr = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
		v2_arr = self.reward*(1-self.g)-self.penalty*self.g-(self.iti+self.g*self.tp)*self.rho
		for i,v_explore in enumerate(self.v_explore()):
			setted_ub = False
			bound1 = 1.
			bound2 = 0.
			for j,(v1_pr,v2_pr,ve_pr,v1,v2,ve) in enumerate(zip(v1_arr[:-1],v2_arr[:-1],v_explore[:-1],v1_arr[1:],v2_arr[1:],v_explore[1:])):
				prev_val_zone = np.argmax([v1_pr,v2_pr,ve_pr])
				curr_val_zone = np.argmax([v1,v2,ve])
				if curr_val_zone==0 and v1==v2:
					bound1 = self.g[j+1]
					bound2 = self.g[j+1]
				elif curr_val_zone==0 and v1==ve:
					bound1 = self.g[j+1]
				elif curr_val_zone==1 and v2==ve:
					bound2 = self.g[j+1]
				elif curr_val_zone!=prev_val_zone:
					if curr_val_zone==2 and prev_val_zone==1:
						bound2 = ((ve-v2)*self.g[j]-(ve_pr-v2_pr)*self.g[j+1]) / (v2_pr-v2+ve-ve_pr)
					elif curr_val_zone==0 and prev_val_zone==2 and not setted_ub:
						setted_ub = True
						bound1 = ((v1-ve)*self.g[j]-(v1_pr-ve_pr)*self.g[j+1]) / (ve_pr-ve+v1-v1_pr)
			self.bounds[:,i] = np.array([bound1,bound2])
		self.bounds[:,-1] = np.array([0.5,0.5])
		return self.bounds
		
	#~ def decision_bounds(self,value=None):
		#~ """
		#~ Compute the decision bounds from the value of the beliefs
		#~ """
		#~ if value is None:
			#~ value = self.value
		#~ self.bounds = np.zeros((2,self.nT))
		#~ v1 = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
		#~ v2 = self.reward*(1-self.g)-self.penalty*self.g-(self.iti+self.g*self.tp)*self.rho
		#~ with np.errstate(invalid='ignore'):
			#~ decide_1 = np.abs((v1-value)/v1)<1e-9
			#~ decide_2 = np.abs((v2-value)/v2)<1e-9
		#~ for i,t in enumerate(self.t):
			#~ if any(decide_1[i]):
				#~ bound1 = self.g[decide_1[i].nonzero()[0][0]]
			#~ else:
				#~ bound1 = 1.
			#~ if any(decide_2[i]):
				#~ bound2 = self.g[decide_2[i].nonzero()[0][-1]]
			#~ else:
				#~ bound2 = 0.
			#~ self.bounds[:,i] = np.array([bound1,bound2])
		#~ return self.bounds
	
	def belief_bound_to_x_bound(self,bounds=None):
		"""
		Transform bounds in belief space to bounds in x(t), i.e.
		diffusing particle's position
		"""
		if bounds is None:
			bounds = self.bounds
		return self.g2x(self.t,bounds)
	
	def belief_bound_to_norm_mu_bound(self,bounds=None):
		"""
		Transform bounds in belief space to bounds in normalized mu
		estimates (mean_mu/std_mu)
		"""
		if bounds is None:
			bounds = self.bounds
		return normcdfinv(bounds)
	
	def belief_bound_to_mu_bound(self,bounds=None):
		"""
		Transform bounds in belief space to bounds in mean mu estimates
		"""
		if bounds is None:
			bounds = self.bounds
		return normcdfinv(bounds)*np.sqrt(self.post_mu_var(self.t))
	
	def refine_value(self,dt=None,n=None):
		"""
		This method re-computes the value of the beliefs using the
		average reward (rho) that was already computed.
		"""
		change = False
		if dt is not None:
			if dt<self.dt:
				oldt = self.t.copy()
				self.t = np.arange(0,self.T+dt,dt,np.float)
				self.nT = self.t.shape[0]
				self.cost = np.interp(self.t, oldt, self.cost)
				change = True
		if n is not None:
			#~ print self.value[0,int(0.5*self.n)]
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
			val0 = self.value[0,int(0.5*self.n)]
			#~ print val0
			if abs(val0)>1e-12:
				# Must refine the value of self.rho
				if val0<0:
					ub = self.rho
					lb = self.rho-1e-2
				else:
					ub = self.rho+1e-2
					lb = self.rho
				self.value_dp(lb,ub)
			self.decision_bounds()
	
	def compute_decision_bounds(self,cost=None,reward=None,n=51,N=501,type_of_bound='belief'):
		if type_of_bound.lower() not in ['belief','norm_mu','mu']:
			raise ValueError("type_of_bound must be 'belief', 'norm_mu' or 'mu'. %s supplied instead."%(type_of_bound))
		self.store_p = True
		if reward is not None:
			self.reward = reward
		self.n = int(n)
		if self.n%2==0:
			self.n+1
		self.g = np.linspace(0.,1.,self.n)
		if cost is not None:
			if np.isscalar(cost):
				self.set_constant_cost(cost)
			else:
				self.cost = cost
		self.invert_belief()
		self.p = self.belief_transition_p()
		self.value_dp()
		del self.p
		self.store_p = False
		self.refine_value(n=N)
		bounds = self.decision_bounds()
		if type_of_bound.lower()=='norm_mu':
			bounds = self.belief_bound_to_norm_mu_bound(bounds)
		elif type_of_bound.lower()=='mu':
			bounds = self.belief_bound_to_mu_bound(bounds)
		return bounds
	
	def rt(self,mu,bounds=None):
		if bounds is None:
			bounds = self.belief_bound_to_x_bound(self.bounds)
		g1 = np.zeros_like(self.t)
		g2 = np.zeros_like(self.t)
		
		for i,t in enumerate(self.t):
			if i==0:
				t0 = t
				continue
			elif i==1:
				g1[i] = -2*self.Psi(mu,bounds[0],i,t,self.prior_mu_mean,t0)
				g2[i] = 2*self.Psi(mu,bounds[1],i,t,self.prior_mu_mean,t0)
			else:
				g1[i] = -2*self.Psi(mu,bounds[0],i,t,self.prior_mu_mean,t0)+\
					2*self.dt*np.sum(g1[:i]*self.Psi(mu,bounds[0],i,t,bounds[0][:i],self.t[:i]))+\
					2*self.dt*np.sum(g2[:i]*self.Psi(mu,bounds[0],i,t,bounds[1][:i],self.t[:i]))
				g2[i] = 2*self.Psi(mu,bounds[1],i,t,self.prior_mu_mean,t0)-\
					2*self.dt*np.sum(g1[:i]*self.Psi(mu,bounds[1],i,t,bounds[0][:i],self.t[:i]))-\
					2*self.dt*np.sum(g2[:i]*self.Psi(mu,bounds[1],i,t,bounds[1][:i],self.t[:i]))
		return g1,g2
		
	def Psi(self,mu,bound,itp,tp,x0,t0):
		normpdf = np.exp(-0.5*(bound[itp]-x0-mu*(tp-t0))**2/self.model_var/(tp-t0))/np.sqrt(2.*np.math.pi*self.model_var*(tp-t0))
		bound_prime = (bound[itp+1]-bound[itp])/self.dt if itp<len(bound)-1 else 0.
		return 0.5*normpdf*(bound_prime-(bound[itp]-x0)/(tp-t0))

def sim_rt(mu,sigma,dt,T,xb,reps=10000):
	sim = np.zeros(reps)
	rt = np.zeros(reps)
	decision = np.zeros(reps)
	not_decided = np.ones(reps)
	for i,t in enumerate(np.arange(float(dt),float(T+dt),float(dt),np.float)):
		sim+= dt*sigma*np.random.randn(reps)
		print np.min(sim),np.max(sim)
		stim = sim+t*mu
		dec1 = np.logical_and(stim>=xb[0][i+1],not_decided)
		dec2 = np.logical_and(stim<=xb[1][i+1],not_decided)
		if any(dec1):
			rt[dec1] = t
			decision[dec1] = 1
			not_decided[dec1] = 0
		if any(dec2):
			rt[dec2] = t
			decision[dec2] = 2
			not_decided[dec2] = 0
		if not any(not_decided):
			break
	return rt,decision


def tesis_figure():
	m = DecisionPolicy(model_var=50/0.04,prior_mu_var=20,n=51,T=100.,\
					   reward=1,cost=5,penalty=0,iti=10.,tp=5.,store_p=False)
	m.refine_value(n=501)
	b = m.bounds
	xb = m.belief_bound_to_x_bound(b)
	io.savemat('tesis_figure_data3',{'value':m.value,'gb':b,'xb':xb,'v_explore':m.v_explore(),'t':m.t,'g':m.g}) 
	
	plt.figure(figsize=(11,8))
	plt.subplot(121)
	plt.imshow(m.value.T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[m.t[0],m.t[-1],m.g[0],m.g[-1]])
	#~ plt.contourf(m.value.T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[m.t[0],m.t[-1],m.g[0],m.g[-1]])
	cbar = plt.colorbar(orientation='horizontal')
	cbar.set_label('Value')
	plt.ylabel('belief')
	plt.xlabel('T [s]')
	plt.subplot(222)
	plt.plot(m.t,b.T)
	plt.ylabel('Belief bound')
	plt.subplot(224)
	plt.plot(m.t,xb.T)
	plt.ylabel('$x(t)$ bound')
	plt.xlabel('T [s]')
	plt.show()

def test():
	m = DecisionPolicy(model_var=50/0.04,prior_mu_var=20,n=51,T=30.,\
					   reward=1,cost=5,penalty=0,iti=10.,tp=5.,store_p=False)
	m.refine_value(n=501)
	b = m.bounds
	xb = m.belief_bound_to_x_bound(b)
	plt.figure(figsize=(11,8))
	plt.subplot(121)
	plt.imshow(m.value.T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[m.t[0],m.t[-1],m.g[0],m.g[-1]])
	#~ plt.contourf(m.value.T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[m.t[0],m.t[-1],m.g[0],m.g[-1]])
	cbar = plt.colorbar(orientation='horizontal')
	cbar.set_label('Value')
	plt.ylabel('belief')
	plt.xlabel('T [s]')
	plt.subplot(222)
	plt.plot(m.t,b.T)
	plt.ylabel('Belief bound')
	plt.subplot(224)
	plt.plot(m.t,xb.T)
	plt.ylabel('$x(t)$ bound')
	plt.xlabel('T [s]')
	
	
	#~ colors = plt.get_cmap('jet')
	#~ plt.gca().set_color_cycle([colors(v) for v in np.linspace(0,1,m.value.shape[0])])
	#~ plt.plot(m.g,m.value.T)
	#~ plt.xlabel('belief')
	#~ plt.ylabel('value')
	#~ plt.subplot(322)
	#~ plt.plot(m.t,b.T)
	#~ plt.ylabel('Belief bound')
	#~ plt.subplot(324)
	#~ plt.plot(m.t,mu_b.T)
	#~ plt.ylabel('$\mu$ bound')
	#~ plt.subplot(326)
	#~ plt.plot(m.t,nmu_b.T)
	#~ plt.ylabel('$\mu/\sigma$ bound')
	#~ plt.xlabel('T [s]')
	plt.show()

if __name__=="__main__":
	#~ test()
	tesis_figure()
