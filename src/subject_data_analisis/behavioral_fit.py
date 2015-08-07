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
import sys, itertools, math, cma

_vectErf = np.vectorize(math.erf,otypes=[np.float])
def normcdf(x,mu=0.,sigma=1.):
	try:
		new_x = (x-mu)/sigma
	except ZeroDivisionError:
		new_x = np.sign(x-mu)*np.inf
	return 0.5 + 0.5*_vectErf(new_x / np.sqrt(2.0))

def dprime_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	return post_mu_t/np.sqrt(post_va_t)-post_mu_d/np.sqrt(post_va_d)

def var_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	return (post_mu_t-post_mu_d)/(post_va_t+post_va_d)

def dprime_var_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	return post_mu_t/post_va_t-post_mu_d/post_va_d

def fit_rt(subject,model,initial_parameters=[1.6,400.],ISI=40.):
	# The parameters that will be fitted are the decision threshold and
	# the dead time in ms after decision
	
	# Load the subject data and compute the subject's RT distribution
	dat,t,d = subject.load_data()
	t = np.mean(t,axis=2)
	d = np.mean(d,axis=2)
	max_rt_ind = int(math.ceil(max(dat[:,1])/ISI))
	subject_rt = dat[:,1]
	
	# Prepare to compute the model RT distribution for a set of
	# syntethic trials
	sigma = 5.
	target = np.zeros((len(dat),max_rt_ind+1))
	distractor = np.zeros((len(dat),max_rt_ind+1))
	target[:,:t.shape[1]] = t
	distractor[:,:d.shape[1]] = d
	target[:,t.shape[1]:] = (sigma*np.random.randn(max_rt_ind+1-t.shape[1],len(dat))+dat[:,0]).T
	distractor[:,d.shape[1]:] = (sigma*np.random.randn(max_rt_ind+1-d.shape[1],len(dat))+50).T
	
	merit = rt_merit
	ubound = np.array([20.,max_rt_ind*ISI])
	# Initial search width
	sigma0 = 1/3
	# Additional cma options
	cma_options = cma.CMAOptions()
	cma_options.set('bounds',[np.zeros(len(ubound)),ubound])
	cma_options.set('scaling_of_variables',ubound)
	return cma.fmin(merit,initial_parameters,sigma0,options=cma_options,args=(subject_rt,model,target,distractor,ISI),restarts=0),np.arange(len(target)),target,distractor,model
	#~ return [initial_parameters],np.arange(len(target)),target,distractor,model

def fit_rt_distribution(subject,model,synthetic_trials=None,initial_parameters=[1.6,400.],ISI=40.):
	# The parameters that will be fitted are the decision threshold and
	# the dead time in ms after decision
	
	# Load the subject data and compute the subject's RT distribution
	dat,t,d = subject.load_data()
	max_rt_ind = int(math.ceil(max(dat[:,1])/ISI))
	bin_edges = np.array(range(max_rt_ind+1))*ISI
	subject_rt = dat[:,1]
	subject_rt_histogram,_ = np.histogram(subject_rt,bin_edges,density=True)
	
	# Prepare to compute the model RT distribution for a set of
	# syntethic trials
	if synthetic_trials:
		targetmean,indeces = io.increase_histogram_count(dat[:,0],synthetic_trials)
	else:
		targetmean = dat[:,0]
		indeces = np.array(range(dat.shape[0]),dtype=np.int)
	distractormean = 50
	sigma = 5.
	target = (np.random.randn(max_rt_ind,targetmean.shape[0])*sigma+targetmean).T
	distractor = (np.random.randn(max_rt_ind,targetmean.shape[0])*sigma+distractormean).T
	
	# Upper bound of parameter search space (lower bound is [0,0])
	if len(initial_parameters)==2:
		merit = rt_histogram_merit
		ubound = np.array([20.,bin_edges[-1]])
	elif len(initial_parameters)==3:
		merit = rt_histogram_merit_variable_deadtime
		ubound = np.array([20.,bin_edges[-1],200.])
	else:
		raise(ValueError("Initial parameters can have length 2 or 3 depending on whether a variable dead time is being fit."))
	# Initial search width
	sigma0 = 1/3
	# Additional cma options
	cma_options = cma.CMAOptions()
	cma_options.set('bounds',[np.zeros(len(ubound)),ubound])
	cma_options.set('scaling_of_variables',ubound)
	#~ return cma.fmin(merit,initial_parameters,sigma0,options=cma_options,args=(subject_rt_histogram,bin_edges,model,target,distractor),restarts=0),indeces,target,distractor,model
	return [initial_parameters],indeces,target,distractor,model

def rt_merit(params,subject_rt,model,target,distractor,ISI):
	model.threshold = params[0]
	simulation = model.batchInference(target,distractor)
	simulated_rt = simulation[:,3]+params[1]
	return np.sum((subject_rt-simulated_rt)**2)

def rt_histogram_merit(params,subject_rt,bin_edges,model,target,distractor):
	model.threshold = params[0]
	simulation = model.batchInference(target,distractor)
	simulated_rt_histogram,_ = np.histogram(simulation[:,3]+params[1],bin_edges,density=True)
	return np.sum((subject_rt_histogram-simulated_rt_histogram)**2)#/(len(bin_edges)-1)

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
	return np.sum((subject_rt_histogram-simulated_rt_histogram)**2)#/(len(bin_edges)-1)

def test(data_dir='/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles'):
	# Load subject data
	subjects = io.unique_subjects(data_dir)
	ms = io.merge_subjects(subjects)
	dat,t,d = ms.load_data()
	ISI = 40.
	
	# Set model to fit the data
	priors={'prior_mu_t':50,'prior_mu_d':50,'prior_va_t':15**2,'prior_va_d':15**2}
	model = pe.KnownVarPerfectInference(model_var_t=25.,model_var_d=25.,ISI=ISI,**priors)
	# By defaul, model criteria is the optimum criteria
	model.criteria = dprime_var_criteria
	
	# Fit model parameters
	# Best fit for variable dead time with optimum criteria [1.9678714549600138, 621.71742870406183, 75.978678076060476]
	# Best fit of rt values with optimum criteria [0.0020047699075050967, 1127.5089122768973]
	# Best fit of rt values with dprime criteria [0.0069443615567393365, 1127.3789676635531]
	initial_parameters = [1.6, 409.2358]
	#~ fit_output,indeces,target,distractor,model = fit_rt(ms,model,initial_parameters=initial_parameters,ISI=ISI)
	fit_output,indeces,target,distractor,model = fit_rt_distribution(ms,model,10000,initial_parameters=initial_parameters,ISI=ISI)
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
	print simulation
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
	print sim_fluctuations.shape
	sim_sdk,sim_sck,sim_sdk_std,sim_sck_std = ke.kernels(fluctuations,simulated_sel,np.ones_like(simulated_sel))
	sim_sT = np.array(range(sim_sdk.shape[1]),dtype=float)*ISI
	
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
