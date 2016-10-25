#!/usr/bin/python
#-*- coding: UTF-8 -*-

from __future__ import division
from __future__ import print_function
import numpy as np
from scipy import io
from scipy import optimize
import math, copy
from utils import normcdf,normcdfinv
try:
	import dp
	use_cpp_extension = True
except:
	use_cpp_extension = False


class DecisionPolicy():
	"""
	Class that implements the dynamic programming method that optimizes
	reward rate and computes the optimal decision bounds
	"""
	def __init__(self,model_var,prior_mu_mean=0.,prior_mu_var=1.,n=500,dt=1e-2,T=10.,\
				 reward=1.,penalty=0.,iti=1.,tp=0.,cost=0.05,store_p=True,discrete_prior=None,\
				 internal_var=0., fixed_stim_duration=0.):
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
		cost = Cost of accumulating new evidence. Can be a float or a numpy array. See set_cost for details of the accepted costs.
		store_p = Boolean indicating whether to store the belief transition probability array (memory expensive!)
		discrete_prior = can be None or a tuple like (mu_prior,weight_prior)
			mu_prior and weight_prior must be numpy 1-D arrays of the same shape
			that hold the discrete set of posible positive mu values (mu_prior) and their
			weight (weight_prior). The initialization then renormalizes the weights to
			0.5, because the prior is assumed symmetric around zero.
		"""
		self.model_var = model_var
		self._model_var = model_var
		if discrete_prior:
			temp = copy.deepcopy(discrete_prior)
			self.mu_prior,self.weight_prior = temp
			self.prior_mu_mean = 0.
			self.weight_prior/=(2*np.sum(self.weight_prior))
			self.prior_mu_var = np.sum(2*self.weight_prior*self.mu_prior**2)
			self.prior_type = 2
		else:
			self.prior_mu_mean = prior_mu_mean
			self.prior_mu_var = prior_mu_var
			self.prior_type = 1
		self.n = int(n)
		if self.n%2==0:
			self.n+1
		self.dg = 1./n;
		self.g = np.linspace(self.dg/2.,1.-self.dg/2.,self.n)
		self.dt = float(dt)
		self.T = float(T)
		self.nT = int(T/dt)+1
		self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
		self.internal_var = internal_var
		self.fixed_stim_duration = fixed_stim_duration
		self.set_cost(cost)
		self.store_p = store_p
		self.reward = reward
		self.penalty = penalty
		self.iti = iti
		self.tp = tp
		self.rho = 0.
	
	def __str__(self):
		if hasattr(self,'_cost_details'):
			_cost_details = self._cost_details
			if _cost_details['type']<2:
				cost = self._cost_details['details']
			else:
				cost = self.cost
		else:
			_cost_details = {'type':None,'details':None}
			cost = self.cost
		if hasattr(self,'bounds'):
			bounds = self.bounds
		else:
			bounds = None
		string = """
<{class_module}.{class_name} object at {address}>
model_var = {model_var}, _model_var = {_model_var}, internal_var = {internal_var},
prior_mu_mean = {prior_mu_mean}, prior_mu_var = {prior_mu_var}, prior_type = {prior_type},
dg = {dg}, n = {n}, dt = {dt}, nT = {nT}, T = {T},
reward = {reward}, penalty = {penalty}, iti = {iti}, tp = {tp}, rho = {rho},
cost_type = {cost_type}, cost = {cost},
use_cpp_extension = {use_cpp_extension},
bounds = {bounds}
		""".format(class_module=self.__class__.__module__,
					class_name=self.__class__.__name__,
					address=hex(id(self)),
					model_var=self.model_var,
					_model_var=self._model_var,
					internal_var=self.internal_var,
					prior_mu_mean=self.prior_mu_mean,
					prior_mu_var=self.prior_mu_var,
					prior_type=self.prior_type,
					dg=self.dg,
					n=self.n,
					dt=self.dt,
					nT=self.nT,
					T=self.T,
					reward=self.reward,
					penalty=self.penalty,
					iti=self.iti,
					tp=self.tp,
					rho=self.rho,
					cost_type=_cost_details['type'],
					cost=cost,
					use_cpp_extension=use_cpp_extension,
					bounds=bounds)
		return string
	
	def set_n(self,n):
		self.n = n
		if self.n%2==0:
			self.n+1
		self.dg = 1./n;
		self.g = np.linspace(self.dg/2.,1.-self.dg/2.,self.n)
	def set_dt(self,dt):
		oldt = self.t
		self.dt = float(dt)
		self.nT = int(self.T/self.dt)+1
		self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
		if self._cost_details['type']==0:
			self.set_constant_cost(self._cost_details['details'])
		elif self._cost_details['type']==1:
			self.set_polynomial_cost(self._cost_details['details'])
		else:
			self.cost = np.interp(self.t[:-1], oldt[:-1], self.cost)
	def set_T(self,T):
		self.T = float(T)
		old_nT = self.nT
		self.nT = int(self.T/self.dt)+1
		self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
		if self._cost_details['type']==0:
			self.set_constant_cost(self._cost_details['details'])
		elif self._cost_details['type']==1:
			self.set_polynomial_cost(self._cost_details['details'])
		else:
			old_cost = self.cost
			self.cost = np.zeros_like(self.t)
			self.cost[:old_nT-1] = old_cost
			self.cost[old_nT-1:] = old_cost[-1]
	
	def reset(self):
		if self.store_p:
			self.p = np.empty(1)
		self.invg = np.empty(1)
		self.value = np.empty(1)
		self.bounds = np.empty(1)
	
	def copy(self):
		out = DecisionPolicy(self.model_var,self.prior_mu_mean,self.prior_mu_var,self.n,self.dt,self.T,\
				 self.reward,self.penalty,self.iti,self.tp,cost=0.05,store_p=self.store_p)
		try:
			out.invg = copy.deepcopy(self.invg)
		except:
			pass
		if self.store_p:
			try:
				out.p = copy.deepcopy(self.p)
			except:
				pass
		try:
			out.value = copy.deepcopy(self.value)
		except:
			pass
		try:
			out.bounds = copy.deepcopy(self.bounds)
		except:
			pass
		try:
			out.cost = copy.deepcoy(self.cost)
			out._cost_details = copy.deepcoy(self._cost_details)
		except:
			pass
		return out
	
	def set_internal_var(self,internal_var):
		self.internal_var = internal_var
		self.model_var = internal_var
		self._model_var = internal_var
	
	def set_cost(self,cost):
		"""
		This function constructs a DecisionPolicy's cost array of shape
		(nT-1,).
		
		Syntax:
		self.set_cost(cost)
		
		Input:
			cost: a float or a numpy ndarray.
			 If cost is a float, self.cost set as a numpy array with all
			of its values equal to the supplied float.
			 If cost is a numpy ndarray, the cost array is constructed
			in one of two ways. If the supplied cost's shape is equal to
			(nT-1,) then the array is copied as is to the DecisionPolicy's
			cost array. If the shape is not equal, then the supplied array
			is assumed to hold the coefficients of a polynomial and
			the cost array is constructed as a polyval(cost,self.t[:-1]).
		
		Related functions:
		set_constant_cost, set_polynomial_cost, set_array_cost
		"""
		if isinstance(cost,np.ndarray) and not np.isscalar(cost):
			s = cost.shape
			if len(s)>1:
				raise ValueError("Cost must be a scalar or a one dimensional numpy ndarray")
			if s[0]==self.nT-1:
				self.set_cost_array(cost)
			else:
				self.set_polynomial_cost(cost)
		else:
			self.set_constant_cost(cost)
	
	def set_constant_cost(self,cost):
		"""
		self.set_constant_cost(cost)
		
		Primitive function that sets the instances cost array as
		self.cost = float(cost)*numpy.ones(self.nT-1)
		"""
		self.cost = float(cost)*np.ones(self.nT-1)
		self._cost_details = {'type':0,'details':cost}
		#~ self.shift_cost()
	
	def set_polynomial_cost(self,coefs):
		"""
		self.set_polynomial_cost(coefs)
		
		Primitive function that sets the instances cost array as
		self.cost = numpy.polyval(coefs,self.t[:-1])
		"""
		self.cost = np.polyval(coefs,self.t[:-1])
		self._cost_details = {'type':1,'details':coefs[:]}
		#~ self.shift_cost()
	
	def set_array_cost(self,cost,shift_cost=False):
		"""
		self.set_array_cost(cost)
		
		Primitive function that sets the instances cost array as
		self.cost = cost[:]
		"""
		self.cost = cost[:]
		self._cost_details = {'type':2,'details':None}
		#~ if shift_cost:
			#~ self.shift_cost()
	
	def shift_cost(self):
		"""
		self.shift_cost()
		
		Shift cost array rigidly until after the fixed_stim_duration
		"""
		index = (self.t>=self.fixed_stim_duration).nonzero()[0][0]
		self.cost[index:] = self.cost[:(self.nT-index)]
		self.cost[:index] = 0.
	
	def post_mu_var(self,t):
		"""
		Bayes update of the posterior variance at time t
		"""
		return 1./(t/self._model_var+1./self.prior_mu_var)
	
	def post_mu_mean(self,t,x):
		"""
		Bayes update of the posterior mean at time t with cumulated sample x
		"""
		return (x/self._model_var+self.prior_mu_mean/self.prior_mu_var)*self.post_mu_var(t)
	
	def x2g(self,t,x):
		"""
		Mapping from cumulated sample x at time t to belief
		"""
		return normcdf(self.post_mu_mean(t,x)/np.sqrt(self.post_mu_var(t)))
	
	def g2x(self,t,g):
		"""
		Mapping from belief at time t to cumulated sample x (inverse of x2g)
		"""
		return self._model_var*(normcdfinv(g)/np.sqrt(self.post_mu_var(t))-self.prior_mu_mean/self.prior_mu_var)
	
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
				p[:,i,j] = -0.5*(invg[i+1]-invg[i,j]-mu_n*self.dt)**2/((post_var[i]*self.dt+self._model_var)*self.dt)+\
						   0.5*(invg[i+1]/self._model_var+self.prior_mu_mean/self.prior_mu_var)**2*post_var[i+1]
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
				p[counter*self.n+j] = -0.5*(invg[i+1]-invg[i,j]-mu_n*self.dt)**2/((post_var[i]*self.dt+self._model_var)*self.dt)+\
						         0.5*(invg[i+1]/self._model_var+self.prior_mu_mean/self.prior_mu_var)**2*post_var[i+1]
		# To avoid overflows, we substract the max value in the numerator and denominator's exponents
		p = (np.exp(p.T-np.max(p,axis=1))/np.sum(np.exp(p.T-np.max(p,axis=1)),axis=0)).T#/(self.g[1]-self.g[0])
		return p
	
	def value_dp(self,lb=-10.,ub=10.):
		"""
		Method that calls the dynamic programming method that computes
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
				m = f(lb)
				M = f(ub)
				if m!=0 and M!=0:
					while np.sign(m)==np.sign(M):
						if (m<M and m<0) or (m>M and m>0):
							lb = ub
							ub = ub*10
							M = f(ub)
						elif (m>M and m<0) or (m<M and m>0):
							ub = lb
							lb = lb*10
							m = f(lb)
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
				p[:,j] = -0.5*(self.invg[i+1]-self.invg[i,j]-mu_n*self.dt)**2/((post_var_t*self.dt+self._model_var)*self.dt)+\
					   0.5*post_var_t1*(self.invg[i+1]/self._model_var+self.prior_mu_mean/self.prior_mu_var)**2
				#~ p[:,j] = 0.5*normcdfinv(self.g)**2*post_var_t1/post_var_t**2-\
						 #~ 0.5*self._model_var/(self._model_var+post_var_t)*(normcdfinv(self.g)/post_var_t1-normcdfinv(g)/post_var_t-mu_n)**2
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
				p[:,j] = -0.5*(self.invg[i+1]-self.invg[i,j]-mu_n*self.dt)**2/((post_var_t*self.dt+self._model_var)*self.dt)+\
					   0.5*post_var_t1*(self.invg[i+1]/self._model_var+self.prior_mu_mean/self.prior_mu_var)**2
			# We transpose the array after the computation so numpy can correctly broadcast the intermediate operations (sum and max)
			p = np.transpose(np.exp(p-np.max(p,axis=0))/np.sum(np.exp(p-np.max(p,axis=0)),axis=0),(1,0))
			v_explore[i] = np.dot(p,self.value[i+1])-(self.cost[i]+self.rho)*dt
			post_var_t1 = post_var_t
		return v_explore
	
	def decision_bounds(self,v_explore_arr=None):
		"""
		Compute the decision bounds from the value of the beliefs
		"""
		if v_explore_arr is None:
			v_explore_arr = self.v_explore()
		self.bounds = np.zeros((2,self.nT))
		v1_arr = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
		v2_arr = self.reward*(1-self.g)-self.penalty*self.g-(self.iti+self.g*self.tp)*self.rho
		for i,v_explore in enumerate(v_explore_arr):
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
	
	if use_cpp_extension:
		def xbounds(self, tolerance=1e-12, set_rho=True, set_bounds=True, return_values=False, root_bounds=None):
			return dp.xbounds(self,tolerance=tolerance, set_rho=set_rho, set_bounds=set_bounds, return_values=return_values, root_bounds=root_bounds)
		xbounds.__doc__ = dp.xbounds.__doc__
		
		def xbounds_fixed_rho(self, rho=None, set_bounds=False, return_values=False):
			return dp.xbounds_fixed_rho(self,rho=rho, set_bounds=set_bounds, return_values=return_values)
		xbounds_fixed_rho.__doc__ = dp.xbounds_fixed_rho.__doc__
		
		def values(self, rho=None):
			return dp.values(self,rho=rho)
		values.__doc__ = dp.values.__doc__
		
		def rt(self,mu,bounds=None):
			return dp.rt(self,mu,bounds=bounds)
		rt.__doc__ = dp.rt.__doc__
	else:
		def xbounds(self, tolerance=1e-12, set_rho=True, set_bounds=True, return_values=False, root_bounds=None):
			"""
			Computes the decision bounds in x(t) space (i.e. the accumulated sensory input space)
			
			(xub, xlb) = xbounds(dp, tolerance=1e-12, set_rho=False, set_bounds=False, return_values=False, root_bounds=None)
			
			(xub, xlb, value, v_explore, v1, v2) = xbounds(dp, ..., return_values=True)
			
			Computes the decision bounds for the decisionPolicy instance.
			'tolerance' is a float that indicates the tolerance when searching for the rho value that yields value[int(n/2)]=0.
			'set_rho' is ignored in this implementation but necesary for the c++ extension.
			'set_bounds' is ignored in this implementation but necesary for the c++ extension.
			If 'return_values' evaluates to True, then the function returns four extra numpy arrays: value, v_explore, v1 and v2. 'value' is an nT by n shaped array that holds the value of a given g at time t. 'v_explore' has shape nT-1 by n and holds the value of exploring at time t with a given g. v1 and v2 are values of immediately deciding for option 1 or 2, and are one dimensional arrays with n elements.
			'root_bounds' must be a tuple of two elements: (lower_bound, upper_bound). Both 'lower_bound' and 'upper_bound' must be floats that represent the lower and upper bounds in which to perform the root finding of rho.
			"""
			self.invert_belief()
			if self.store_p:
				self.belief_transition_p()
			if root_bounds is None:
				root_bounds=(-10.,10.)
			lower_bound,upper_bound = root_bounds
			self.value_dp(lb=lower_bound,ub=upper_bound)
			xb = self.belief_bound_to_x_bound()
			xub = xb[0]
			xlb = xb[1]
			output = (xub,xlb)
			if return_value:
				value = self.value
				v_explore = self.v_explore()
				v1 = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
				v2 = self.reward*(1-self.g)-self.penalty*self.g-(self.iti+self.g*self.tp)*self.rho
				output+=(value,v_explore,v1,v2)
			return output
		
		def xbounds_fixed_rho(self, rho=None, set_bounds=False, return_values=False):
			"""
			Computes the decision bounds in x(t) space (i.e. the accumulated sensory input space) without iterating the value of rho
			
			(xub, xlb) = xbounds(dp, rho=None, set_bounds=False, return_values=False)
			
			(xub, xlb, value, v_explore, v1, v2) = xbounds(dp, ..., return_values=True)
			
			Computes the decision bounds for a decisionPolicy instance specified in 'dp' for a given rho value.
			'rho' is the fixed reward rate value used to compute the decision bounds and values. If rho=None, then the DecisionPolicy instance's rho is used.
			'set_bounds' must be an expression whose 'truthness' can be evaluated. If set_bounds is True, the python DecisionPolicy object's ´bounds´ attribute will be set to the upper and lower bounds in g space computed in the c++ instance. If false, it will do nothing.
			If 'return_values' evaluates to True, then the function returns four extra numpy arrays: value, v_explore, v1 and v2. 'value' is an nT by n shaped array that holds the value of a given g at time t. 'v_explore' has shape nT-1 by n and holds the value of exploring at time t with a given g. v1 and v2 are values of immediately deciding for option 1 or 2, and are one dimensional arrays with n elements.
			"""
			self.invert_belief()
			if store_p:
				self.belief_transition_p()
				self.backpropagate_value(rho=rho)
			else:
				self.memory_efficient_backpropagate_value(rho=rho)
			self.decision_bounds()
			xb = self.belief_bound_to_x_bound()
			xub = xb[0]
			xlb = xb[1]
			output = (xub,xlb)
			if return_value:
				value = self.value
				v_explore = self.v_explore()
				v1 = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
				v2 = self.reward*(1-self.g)-self.penalty*self.g-(self.iti+self.g*self.tp)*self.rho
				output+=(value,v_explore,v1,v2)
			return output
	
		def values(self, rho=None):
			"""
			Computes the values for a given reward rate, rho, and decisionPolicy parameters.
			(value, v_explore, v1, v2) = values(rho=None)
			
			Computes the value for a given belief g as a function of time for a supplied reward rate, rho. If rho is set to None, then the decisionPolicy instance's rho attribute will be used.
			The function returns a tuple of four numpy arrays: value, v_explore, v1 and v2. 'value' is an nT by n shaped array that holds the value of a given g at time t. 'v_explore' has shape nT-1 by n and holds the value of exploring at time t with a given g. v1 and v2 are values of immediately deciding for option 1 or 2, and are one dimensional arrays with n elements.
			"""
			self.invert_belief()
			if store_p:
				self.belief_transition_p()
				self.backpropagate_value(rho=rho)
			else:
				self.memory_efficient_backpropagate_value(rho=rho)
			value = self.value
			v_explore = self.v_explore()
			v1 = self.reward*self.g-self.penalty*(1-self.g)-(self.iti+(1-self.g)*self.tp)*self.rho
			v2 = self.reward*(1-self.g)-self.penalty*self.g-(self.iti+self.g*self.tp)*self.rho
			return (value,v_explore,v1,v2)
		
		def rt(self,mu,bounds=None):
			"""
			Computes the rt distribution for a given drift rate, mu, decisionPolicy parameters and decision bounds in x space, bounds.
			
			(g1, g2) = values(dp,mu,bounds=None)
			
			Computes the rt distribution for selecting options 1 or 2 for a given drift rate mu and the parameters in the decisionPolicy instance dp. If bounds is None, then xbounds is called with default parameters to compute the decision bounds in x space. To avoid this, supply a tuple (xub,xlb) as the one that is returned by the function xbounds. xub and xlb must be one dimensional numpy arrays with the same elements as dp.t.
			"""
			if bounds is None:
				xub,xlb = self.xbounds(set_rho=True, set_bounds=True)
			else:
				xub,xlb = bounds
			g1 = np.zeros_like(self.t)
			g2 = np.zeros_like(self.t)
			bounds_touched = False
			
			for i,t in enumerate(self.t):
				if bounds_touched:
					g1[i] = 0.
					g2[i] = 0.
				else:
					if i==0:
						t0 = t
						continue
					elif i==1:
						g1[i] = -2*self.Psi(mu,xub,i,t,self.prior_mu_mean,t0)
						g2[i] = 2*self.Psi(mu,xlb,i,t,self.prior_mu_mean,t0)
					else:
						g1[i] = -2*self.Psi(mu,xub,i,t,self.prior_mu_mean,t0)+\
							2*self.dt*np.sum(g1[:i]*self.Psi(mu,xub,i,t,xub[:i],self.t[:i]))+\
							2*self.dt*np.sum(g2[:i]*self.Psi(mu,xub,i,t,xlb[:i],self.t[:i]))
						g2[i] = 2*self.Psi(mu,xlb,i,t,self.prior_mu_mean,t0)-\
							2*self.dt*np.sum(g1[:i]*self.Psi(mu,xlb,i,t,xub[:i],self.t[:i]))-\
							2*self.dt*np.sum(g2[:i]*self.Psi(mu,xlb,i,t,xlb[:i],self.t[:i]))
					if g1[i]<0:
						g1[i] = 0.
					if g2[i]<0:
						g2[i] = 0.
				if xub[i]<=xlb:
					bounds_touched = True
			# Normalize probability density
			normalization = np.sum(g1+g2)*self.dt
			g1/=normalization
			g2/=normalization
			return g1,g2
	
	if use_cpp_extension:
		def refine_value(self,tolerance=1e-12,dt=None,n=None,T=None):
			"""
			This method re-computes the value of the beliefs using the
			average reward (rho) that was already computed.
			"""
			change = False
			if dt is not None:
				if dt<self.dt:
					change = True
					oldt = self.t.copy()
					self.dt = float(dt)
					self.nT = int(self.T/self.dt)+1
					self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
					if self._cost_details['type']==0:
						self.set_constant_cost(self._cost_details['details'])
					elif self._cost_details['type']==1:
						self.set_polynomial_cost(self._cost_details['details'])
					else:
						self.cost = np.interp(self.t[:-1], oldt[:-1], self.cost)
			if T is not None:
				if T>self.T:
					change = True
					self.T = float(T)
					old_nT = self.nT
					old_cost = self.cost
					self.nT = int(self.T/self.dt)+1
					self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
					if self._cost_details['type']==0:
						self.set_constant_cost(self._cost_details['details'])
					elif self._cost_details['type']==1:
						self.set_polynomial_cost(self._cost_details['details'])
					else:
						self.cost = np.zeros_like(self.t)
						self.cost[:old_nT-1] = old_cost
						self.cost[old_nT-1:] = old_cost[-1]
			if n is not None:
				n = int(n)
				if n%2==0:
					n+=1
				if n>self.n:
					change = True
					self.n = n
					self.g = np.linspace(0.,1.,self.n)
			if change:
				temp = self.xbounds_fixed_rho(set_bounds=True, return_values=True)
				val0 = temp[2][0,int(0.5*self.n)]
				if abs(val0)>tolerance:
					if val0<0:
						ub = self.rho
						lb = self.rho-1e-2
					else:
						ub = self.rho+1e-2
						lb = self.rho
					xbs = self.xbounds(tolerance=tolerance, set_rho=True, set_bounds=True, root_bounds=(lb,ub))
				else:
					self.value = temp[0]
					xbs = (temp[0],temp[1])
			else:
				xbs = self.belief_bound_to_x_bound()
				xbs = (xbs[0],xbs[1])
			return xbs
	else:
		def refine_value(self,tolerance=1e-12,dt=None,n=None,T=None):
			"""
			This method re-computes the value of the beliefs using the
			average reward (rho) that was already computed.
			"""
			change = False
			if dt is not None:
				if dt<self.dt:
					change = True
					oldt = self.t.copy()
					self.dt = float(dt)
					self.nT = int(self.T/self.dt)+1
					self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
					if self._cost_details['type']==0:
						self.set_constant_cost(self._cost_details['details'])
					elif self._cost_details['type']==1:
						self.set_polynomial_cost(self._cost_details['details'])
					else:
						self.cost = np.interp(self.t[:-1], oldt[:-1], self.cost)
			if T is not None:
				if T>self.T:
					change = True
					self.T = float(T)
					old_nT = self.nT
					self.nT = int(self.T/self.dt)+1
					self.t = np.arange(0.,self.nT,dtype=np.float64)*self.dt
					if self._cost_details['type']==0:
						self.set_constant_cost(self._cost_details['details'])
					elif self._cost_details['type']==1:
						self.set_polynomial_cost(self._cost_details['details'])
					else:
						old_cost = self.cost
						self.cost = np.zeros_like(self.t)
						self.cost[:old_nT-1] = old_cost
						self.cost[old_nT-1:] = old_cost[-1]
			if n is not None:
				n = int(n)
				if n%2==0:
					n+=1
				if n>self.n:
					change = True
					self.n = n
					self.g = np.linspace(0.,1.,self.n)
			if change:
				self.invert_belief()
				if self.store_p:
					self.p = self.belief_transition_p()
					self.backpropagate_value(rho=self.rho)
				else:
					self.memory_efficient_backpropagate_value(rho=self.rho)
				val0 = self.value[0,int(0.5*self.n)]
				if abs(val0)>tolerance:
					# Must refine the value of self.rho
					if val0<0:
						ub = self.rho
						lb = self.rho-1e-2
					else:
						ub = self.rho+1e-2
						lb = self.rho
					self.value_dp(lb,ub)
				self.decision_bounds()
			xb = self.belief_bound_to_x_bound()
			return (xb[0],xb[1])
	
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
	
	def Psi(self,mu,bound,itp,tp,x0,t0):
		normpdf = 0.3989422804014327**np.exp(-0.5*(bound[itp]-x0-mu*(tp-t0))**2/self._model_var/(tp-t0))/(self._model_var*(tp-t0))
		#~ bound_prime = (bound[itp+1]-bound[itp])/self.dt if itp<len(bound)-1 else 0.
		bound_prime = 0.5*(bound[itp+1]-bound[itp-1])/self.dt if itp<len(bound)-1 else 0.
		return 0.5*normpdf*(bound_prime-(bound[itp]-x0)/(tp-t0)-mu)
	
	def rt_nlog_like(self,g,rt):
		"""
		Syntax
		nLL = self.rt_nlog_like(g,rt)
		
		Computes the negative log likelihood (nLL) of response time rt given
		an array of the response time density (g) such as the array
		returned by the method rt. g must be an array with nT elements.
		This function does a linear interpolation of the g array values
		to approximate the nLL for rt's that are not equal to any members
		of array self.t
		"""
		if rt>self.T:
			raise ValueError("Response time: %f is beyond max time T=%f"%(rt,self.T))
		elif rt<0:
			raise ValueError("Response time: %f cannot be below 0"%(rt))
		if rt==self.T:
			return -np.log(g[self.nT-1])
		t_i = int(rt/self.dt)
		if (rt-self.t[t_i])==0:
			return -np.log(g[t_i])
		ret = -np.log(g[t_i]+(g[t_i+1]-g[t_i])/self.dt*(rt-self.t[t_i]))
		return ret
	
	def log_odds(self):
		ret = np.log(self.bounds)-np.log(1-self.bounds)
		ret[1]*=-1
		return ret

def diffusion_path_samples(mu,var_rate,dt,T,xb,reps=10):
	paths = []
	sigma = np.sqrt(var_rate*dt)
	if not isinstance(mu,np.ndarray):
		mus = mu*np.ones(reps)
	else:
		mus = mu
	nT = int(T/dt)+1
	for mu in mus:
		path = {'x':[0],'t':[0]}
		decided = False
		for t_i in np.arange(1,nT):
			t = t_i*dt
			stim = sigma*np.random.randn(1)+mu*dt
			path['x'].append(path['x'][-1]+stim)
			path['t'].append(t)
			if path['x'][-1]>=xb[0][t_i+1]:
				path['dec']=1
				path['rt']=t
				decided = True
				break
			elif path['x'][-1]<=xb[1][t_i+1]:
				path['dec']=2
				path['rt']=t
				decided = True
				break
		if not decided:
			path['dec']=None
			path['rt']=None
		paths.append(path)
	return sorted(paths,key=lambda path:path['rt'], reverse=True)

def sim_rt(mu,var_rate,dt,T,xb,reps=10000):
	if not isinstance(mu,np.ndarray):
		mu = mu*np.ones(reps)
	sim = np.zeros_like(mu)
	rt = np.zeros_like(mu)
	decision = np.zeros_like(mu)
	not_decided = np.ones_like(mu,dtype=np.bool)
	sigma = np.sqrt(dt*var_rate)
	nT = int(T/dt)+1
	for i in range(1,nT):
		t = float(dt)*i
		sim+= sigma*np.random.randn(*mu.shape)
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
	out = (rt,decision)
	return out

def main():
	from matplotlib import pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	def bounds_for_costs(n=20,maxcost=1.):
		m = DecisionPolicy(model_var=50/0.04,prior_mu_var=62500,n=101,T=10,dt=0.04,reward=1,cost=1./5.,penalty=0,iti=10.,tp=5.,store_p=False)
		costs = np.linspace(0,maxcost,n)
		s1_colors = [plt.get_cmap('YlGn')(x) for x in np.linspace(1,0,n)]
		s2_colors = [plt.get_cmap('YlOrRd')(x) for x in np.linspace(1,0,n)]
		
		for cost,c1,c2 in zip(costs,s1_colors,s2_colors):
			m.cost = cost
			xub,xlb = m.xbounds()
			plt.plot(m.t,xub,color=c1)
			plt.plot(m.t,xlb,color=c2)
		
		pp = PdfPages('../../figs/bounds_cost.pdf')
		pp.savefig()
		pp.close()
		
		plt.show()


	def tesis_figure():
		m = DecisionPolicy(model_var=50/0.04,prior_mu_var=62500,n=101,T=50,dt=0.01,reward=1,penalty=0,iti=10.,tp=5.)
		m.cost = 0.1
		m.xbounds()
		m.refine_value(n=501)
		xub,xlb,value,v_explore,v1,v2 = m.xbounds_fixed_rho(set_bounds=True,return_values=True)
		xb = np.array([xub,xlb])
		b = m.bounds
		print(xb.shape, b.shape)
		io.savemat('tesis_figure_data4',{'value':value,'gb':b,'xb':xb,'v_explore':v_explore,'v1':v1,'v2':v2,'t':m.t,'g':m.g}) 
		
		plt.figure(figsize=(11,8))
		plt.subplot(221)
		plt.imshow(value.T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[m.t[0],m.t[-1],m.g[0],m.g[-1]])
		#~ plt.contourf(m.value.T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[m.t[0],m.t[-1],m.g[0],m.g[-1]])
		cbar = plt.colorbar(orientation='vertical')
		cbar.set_label('Value')
		plt.ylabel('belief')
		plt.xlabel('T [s]')
		
		plt.subplot(223)
		plt.plot(m.g,np.max(np.array([v1,v2]),axis=0),linestyle='--',color='k',linewidth=3)
		inds = np.array([0,100,250,500,len(m.t)-2])
		cmap = plt.get_cmap('RdYlGn')
		for ind,cmap_ind in zip(inds,1-inds.astype(np.float64)/float((len(m.t)-1)*10./m.T)):
			plt.plot(m.g,v_explore[ind],color=cmap(cmap_ind),linewidth=2)
		plt.xlabel('Belief')
		plt.ylabel('$\tilde V$')
		
		plt.subplot(222)
		plt.plot(m.t,b.T)
		plt.ylabel('Belief bound')
		plt.subplot(224)
		plt.plot(m.t,xb.T)
		plt.ylabel('$x(t)$ bound')
		plt.xlabel('T [s]')
		
		pp = PdfPages('../../figs/inference_value.pdf')
		pp.savefig()
		pp.close()
		
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
	
	
	#~ test()
	#~ tesis_figure()
	bounds_for_costs()

if __name__=="__main__":
	main()
