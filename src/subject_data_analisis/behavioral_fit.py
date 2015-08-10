#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package that implements perfect inference behavior """

from __future__ import division
import numpy as np
import kernels as ke
import data_io as io
import perfect_inference as pe
try:
	import matplotlib as mt
	from matplotlib import pyplot as plt
	interactive_pyplot = plt.isinteractive()
	loaded_plot_libs = True
except:
	loaded_plot_libs = False
import sys, itertools, math, cma, os, pickle

_vectErf = np.vectorize(math.erf,otypes=[np.float])
def normcdf(x,mu=0.,sigma=1.):
	try:
		new_x = (x-mu)/sigma
	except ZeroDivisionError:
		new_x = np.sign(x-mu)*np.inf
	return 0.5 + 0.5*_vectErf(new_x / np.sqrt(2.0))

def optimal_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	return (post_mu_t-post_mu_d)/np.sqrt(post_va_t+post_va_d)

def dprime_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	return post_mu_t/np.sqrt(post_va_t)-post_mu_d/np.sqrt(post_va_d)

def var_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	return (post_mu_t-post_mu_d)/(post_va_t+post_va_d)

def dprime_var_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	return post_mu_t/post_va_t-post_mu_d/post_va_d

def rt_merit(params,subject_rt,model,target,distractor,ISI):
	model.threshold = params[0]
	simulation = model.batchInference(target,distractor)
	simulated_rt = simulation[:,3]+params[1]
	return np.sum((subject_rt-simulated_rt)**2)

def rt_histogram_merit(params,subject_rt,bin_edges,model,target,distractor):
	model.threshold = params[0]
	simulation = model.batchInference(target,distractor)
	simulated_rt_histogram,_ = np.histogram(simulation[:,3]+params[1],bin_edges,density=True)
	return np.sum((subject_rt_histogram-simulated_rt_histogram)**2)

def rt_histogram_merit_variable_deadtime(params,subject_rt_histogram,bin_edges,model,target,distractor):
	model.threshold = params[0]
	simulation = model.batchInference(target,distractor)
	simulated_rt_histogram,_ = np.histogram(simulation[:,3]+params[1],bin_edges,density=True)
	# Include a normal fluctuation to the simulated_rt with width params[2]
	# by convolving the simulated rt with a gaussian pdf
	conv_window = np.linspace(-params[2]*6,params[2]*6,int(math.ceil(params[2]*12/model.ISI))+1)
	conv_val = np.zeros_like(conv_window)
	conv_val[1:-1] = np.diff(normcdf(conv_window[:-1],0,params[2]))
	conv_val[0] = normcdf(conv_window[0],0,params[2])
	conv_val[-1] = 1-normcdf(conv_window[-2],0,params[2])
	simulated_rt_histogram = np.convolve(simulated_rt_histogram,conv_val,mode='same')
	return np.sum((subject_rt_histogram-simulated_rt_histogram)**2)

def load_fitter(filename):
	if not filename.endswith('.pkl'):
		filename+='.pkl'
	f = open(filename,'r')
	output = pickle.load(f)
	f.close()
	return output

class Fitter():
	def __init__(self,subject,model=None,objective='rt distribution',initial_parameters=None,ubounds=None,\
				criteria=None,synthetic_trials=None,cma_sigma=1/3,cma_options=cma.CMAOptions(),use_subject_signals=False):
		# Subject data
		self.subject = subject
		self.use_subject_signals = use_subject_signals
		# Model
		if model:
			self.model = model
		else:
			self.model = pe.UnknownVarPerfectInference()
			self.model.ISI = 40.
		if criteria:
			if criteria.lower()=='optimal':
				self.model.criteria = optimal_criteria
			elif criteria.lower()=='dprime':
				self.model.criteria = dprime_criteria
			elif criteria.lower()=='dprime_var':
				self.model.criteria = dprime_var_criteria
			elif criteria.lower()=='var':
				self.model.criteria = var_criteria
			else:
				raise(ValueError("Unknown criteria %s",criteria))
		else:
			# Use the supplied model's criteria
			pass
		# Objective function (merit)
		if objective.lower() in ('rt distribution','rt histogram','rt_distribution','rt_histogram'):
			self.fittype = 0
			if initial_parameters:
				self.initial_parameters = initial_parameters
			else:
				self.initial_parameters = np.random.random(2)
			if len(self.initial_parameters)==2:
				self.merit = rt_histogram_merit
			else:
				self.merit = rt_histogram_merit_variable_deadtime
			if self.use_subject_signals and synthetic_trials:
				raise(Warning("Variable synthetic_trials is not used when use_subject_signals is True"))
				self.synthetic_trials = None
			else:
				self.synthetic_trials = synthetic_trials
		elif objective.lower()=='rt':
			if len(initial_parameters)!=2:
				raise(ValueError("If the objective is to fit the RT values, only the model threshold and dead time can be fitted, not the dead time dispersion. Thus, initial_parameters must have only two elements"))
			self.fittype = 1
			self.synthetic_trials = None
			if synthetic_trials:
				raise(Warning("Variable synthetic_trials is not used when the objective is to fit the RT values"))
			self.initial_parameters = initial_parameters
			self.merit = rt_merit
		else:
			raise(ValueError("Unknown fit objective: %s",objective))
		# CMA parameters
		self.cma_sigma = cma_sigma
		self.cma_options = cma_options
		if ubounds:
			if len(ubounds)!=len(initial_parameters):
				raise(ValueError("Parameter upper boundary must be an array with the same length as the supplied initial_parameters"))
			self.cma_options.set('bounds',[np.zeros(len(initial_parameters)),ubounds])
			self.cma_options.set('scaling_of_variables',ubounds)
		
		# Fitting output
		self.synthetic_data = {}
		self.fit_ouput = {}
	
	def fit(self,restarts=0,**kwargs):
		# Load subject data
		dat,t,d = self.subject.load_data()
		max_rt_ind = int(math.ceil(max(dat[:,1])/self.model.ISI))
		bin_edges = np.array(range(max_rt_ind+1))*self.model.ISI
		subject_rt = dat[:,1]
		# Construct synthetic data for parameter fitting
		if self.synthetic_trials:
			n_trials = self.synthetic_trials
			targetmean,indeces = io.increase_histogram_count(dat[:,0],self.synthetic_trials)
		else:
			n_trials = dat.shape[0]
			targetmean = dat[:,0]
			indeces = np.array(range(n_trials),dtype=np.int)
		sigma = 5.
		target = np.zeros((n_trials,max_rt_ind+1))
		distractor = np.zeros((n_trials,max_rt_ind+1))
		if self.use_subject_signals:
			target[:,:t.shape[1]] = t
			distractor[:,:d.shape[1]] = d
			target[:,t.shape[1]:] = (sigma*np.random.randn(max_rt_ind+1-t.shape[1],n_trials)+dat[:,0]).T
			distractor[:,d.shape[1]:] = (sigma*np.random.randn(max_rt_ind+1-d.shape[1],n_trials)+50).T
		else:
			target = (sigma*np.random.randn(max_rt_ind+1,n_trials)+dat[indeces,0]).T
			distractor = (sigma*np.random.randn(max_rt_ind+1,n_trials)+50).T
		self.synthetic_data = {'target':target,'distractor':distractor,'indeces':indeces}
		if self.fittype==0:
			subject_rt_histogram,_ = np.histogram(subject_rt,bin_edges,density=True)
			args = (subject_rt_histogram,bin_edges,self.model,target,distractor)
		elif self.fittype==1:
			args = (subject_rt,self.model,target,distractor,self.model.ISI)
		else:
			args = (None,)
		#~ self.fit_output = cma.fmin(self.merit,self.initial_parameters,self.cma_sigma,options=self.cma_options,args=args,restarts=restarts,**kwargs)
		self.fit_output = (self.initial_parameters,)
		return self.fit_output
	
	def __dict__(self):
		return {'subject':self.subject,'model':self.model,'use_subject_signals':self.use_subject_signals,\
				'fittype':self.fittype,'synthetic_trials':self.synthetic_trials,'initial_parameters':self.initial_parameters,\
				'merit':self.merit,'cma_sigma':self.cma_sigma,'cma_options':self.cma_options,\
				'synthetic_data':self.synthetic_data,'fit_ouput':self.fit_ouput}
	
	def save(self,filename,overwrite=False):
		if not filename.endswith('.pkl'):
			filename+='.pkl'
		if os.path.isfile(filename) and not overwrite:
			raise(IOError("Supplied file %s already exits. If use wish to overwrite it, call save(...,overwrite=True) instead",filename))
		f = open(filename,'w')
		pickle.dump(self,f)
		f.close()

def test(data_dir='/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles'):
	# Load subject data
	subjects = io.unique_subjects(data_dir)
	ms = io.merge_subjects(subjects)
	dat,t,d = ms.load_data()
	ISI = 40.
	
	# Set model to fit the data
	priors={'prior_mu_t':50,'prior_mu_d':50,'prior_va_t':15**2,'prior_va_d':15**2}
	model = pe.KnownVarPerfectInference(model_var_t=25.,model_var_d=25.,ISI=ISI,**priors)
	initial_parameters = [1.9678714549600138, 621.71742870406183, 75.978678076060476]
	ubounds = [20.,max(dat[:,1]),200.]
	fitter = Fitter(ms,model,objective=u'rt distribution',initial_parameters=initial_parameters,ubounds=ubounds,\
				criteria=None,synthetic_trials=10000,cma_sigma=1/3,cma_options=cma.CMAOptions(),use_subject_signals=False)
	fitter.save('Fit_test')
	
	fit_output = fitter.fit(restarts=0)
	target = fitter.synthetic_data['target']
	distractor = fitter.synthetic_data['distractor']
	indeces = fitter.synthetic_data['indeces']
	
	if len(fit_output[0])==3:
		var_dead_time=fit_output[0][2]
	else:
		var_dead_time = None
	
	# Compute subject rt distribution (only for plotting purposes)
	
	max_rt_ind = int(math.ceil(max(dat[:,1])/ISI))
	bin_edges = np.array(range(max_rt_ind+1))*ISI
	subject_rt,bin_edges = np.histogram(dat[:,1],bin_edges,density=True)
	# Compute subject kernels (only for plotting purposes)
	t = (np.mean(t,axis=2,keepdims=False).T-dat[:,0]).T
	d = np.mean(d,axis=2,keepdims=False)-50
	fluctuations = np.transpose(np.array([t,d]),(1,0,2))
	sdk,sck,sdk_std,sck_std = ke.kernels(fluctuations,1-dat[:,2],dat[:,3]-1)
	sT = np.array(range(sdk.shape[1]),dtype=float)*ISI
	
	# Compute fitted model's rt distribution
	model.threshold = fit_output[0][0]
	simulation = model.batchInference(target,distractor)
	simulated_rt,_ = np.histogram(simulation[:,3]+fit_output[0][1],bin_edges,density=True)
	if var_dead_time:
		conv_window = np.linspace(-var_dead_time*6,var_dead_time*6,int(math.ceil(var_dead_time*12/model.ISI))+1)
		conv_val = np.zeros_like(conv_window)
		conv_val[1:-1] = np.diff(normcdf(conv_window[:-1],0,var_dead_time))
		conv_val[0] = normcdf(conv_window[0],0,var_dead_time)
		conv_val[-1] = 1-normcdf(conv_window[-2],0,var_dead_time)
		simulated_rt = np.convolve(simulated_rt,conv_val,mode='same')
	
	print "Fraction of decided trials: ",np.sum(simulation[:,0])/simulation.shape[0]
	print "Performance in decided trials: ",np.nanmean(simulation[:,1])
	simulated_sel = 1-simulation[:,1]
	simulated_sel[np.isnan(simulated_sel)] = 1
	# Compute fitted model's kernels
	sim_fluctuations = np.transpose(np.array([target.T-dat[indeces,0],distractor.T-50]),(2,0,1))
	sim_sdk,sim_sck,sim_sdk_std,sim_sck_std = ke.kernels(sim_fluctuations,simulated_sel,np.ones_like(simulated_sel))
	sim_sT = np.array(range(sim_sdk.shape[1]),dtype=float)*ISI
	
	sim_sT[sdk.shape[1]:] = np.nan
	sim_sdk[:,sdk.shape[1]:] = np.nan
	sim_sck[:,sdk.shape[1]:] = np.nan
	sim_sdk_std[:,sdk.shape[1]:] = np.nan
	sim_sck_std[:,sdk.shape[1]:] = np.nan
	
	if loaded_plot_libs:
		if not interactive_pyplot:
			plt.ioff() # cma overrides original pyplot settings and if interactive mode was off, the plot is never displayed!
		plt.figure(figsize=(13,10))
		plt.subplot(121)
		plt.plot(bin_edges[:-1],subject_rt,'k')
		plt.plot(bin_edges[:-1],simulated_rt,'r')
		plt.legend(['Subject','Simulation'])
		plt.xlabel('Response time [ms]')
		plt.ylabel('Prob density [1/ms]')
		plt.subplot(122)
		plt.plot(sT,sdk[0],color='b',linestyle='--')
		plt.plot(sT,sdk[1],color='r',linestyle='--')
		plt.plot(sim_sT,sim_sdk[0],color='b')
		plt.plot(sim_sT,sim_sdk[1],color='r')
		plt.fill_between(sT,sdk[0]-sdk_std[0],sdk[0]+sdk_std[0],color='b',alpha=0.3,edgecolor=None)
		plt.fill_between(sT,sdk[1]-sdk_std[1],sdk[1]+sdk_std[1],color='r',alpha=0.3,edgecolor=None)
		plt.fill_between(sim_sT,sim_sdk[0]-sim_sdk_std[0],sim_sdk[0]+sim_sdk_std[0],color='b',alpha=0.3,edgecolor=None)
		plt.fill_between(sim_sT,sim_sdk[1]-sim_sdk_std[1],sim_sdk[1]+sim_sdk_std[1],color='r',alpha=0.3,edgecolor=None)
		plt.plot(plt.gca().get_xlim(),[0,0],color='k')
		plt.xlabel('Time [ms]')
		plt.ylabel('Fluctuation [$cd/m^{2}$]')
		plt.legend(['Subject $D_{S}$','Subject $D_{N}$','Simulated $D_{S}$','Simulated $D_{N}$'])
		plt.show()
	return 0

if __name__=="__main__":
	if len(sys.argv)>1:
		test(sys.argv[1])
	else:
		test()
