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
import sys, itertools, math, cma, os, pickle, warnings

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

def conv_hist(h,p,ISI,mode='full'):
	conv_window = np.linspace(-p*6,p*6,int(math.ceil(p*12/ISI))+1)
	conv_val = np.zeros_like(conv_window)
	conv_val[1:-1] = np.diff(normcdf(conv_window[:-1],0,p))
	conv_val[0] = normcdf(conv_window[0],0,p)
	conv_val[-1] = 1-normcdf(conv_window[-2],0,p)
	ch = np.convolve(h,conv_val,mode=mode)
	a = int(0.5*(ch.shape[0]-h.shape[0]))
	if a==0:
		if 0.5*(ch.shape[0]-h.shape[0])==0:
			ret = ch
		else:
			ret = ch[:-1]
	elif a==0.5*(ch.shape[0]-h.shape[0]):
		ret = ch[a:-a]
	else:
		ret = ch[a:-a-1]
	return ret

def optimal_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	"""
	Optimal decision criteria for discriminating which of two gaussians
	has higher mean. (mu_t-mu_d)/sqrt(var_t+var_d)
	"""
	return (post_mu_t-post_mu_d)/np.sqrt(post_va_t+post_va_d)

def dprime_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	"""
	Dprime decision criteria for discriminating which of two gaussians
	has higher mean. mu_t/sqrt(var_t)-mu_d/sqrt(var_d)
	"""
	return post_mu_t/np.sqrt(post_va_t)-post_mu_d/np.sqrt(post_va_d)

def var_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	"""
	Modification of optimal decision criteria for discriminating which
	of two gaussians has higher mean. (mu_t-mu_d)/(var_t+var_d)
	"""
	return (post_mu_t-post_mu_d)/(post_va_t+post_va_d)

def dprime_var_criteria(post_mu_t,post_mu_d,post_va_t,post_va_d):
	"""
	Modification of dprime decision criteria for discriminating which of
	two gaussians has higher mean. mu_t/var_t-mu_d/var_d
	"""
	return post_mu_t/post_va_t-post_mu_d/post_va_d

def rt_merit(params,subject_rt,model,target,distractor,ISI):
	"""
	Compute least square difference between subject and simulation
	response times. This assumes that the subject and the model viewed
	exactly the same stimuli, allowing trial to trial comparison of RT.
	"""
	model.threshold = params[0]
	simulation = model.batchInference(target,distractor)
	simulated_rt = simulation[:,3]+params[1]
	return np.sum((subject_rt-simulated_rt)**2)

def rt_histogram_merit(params,subject_rt,bin_edges,model,target,distractor):
	"""
	Compute least square difference between subject and simulation
	response time histograms. This only compares the histograms of RT's
	for the model and the subjects, it does not compare differences
	between response time pairs as rt_merit. This function can be used
	even when the model and the subject did not see the same stimuli.
	"""
	model.threshold = params[0]
	simulation = model.batchInference(target,distractor)
	simulated_rt_histogram,_ = np.histogram(simulation[:,3]+params[1],bin_edges,density=True)
	return np.sum((subject_rt_histogram-simulated_rt_histogram)**2)

def rt_histogram_merit_variable_deadtime(params,subject_rt_histogram,bin_edges,model,target,distractor):
	"""
	The same as rt_histogram but allowing for random time shifts added
	to the model decision times.
	"""
	model.threshold = params[0]
	simulation = model.batchInference(target,distractor)
	simulated_rt_histogram,_ = np.histogram(simulation[:,3]+params[1],bin_edges,density=True)
	# Include a normal fluctuation to the simulated_rt with width params[2]
	# by convolving the simulated rt with a gaussian pdf
	simulated_rt_histogram = conv_hist(simulated_rt_histogram,params[2],model.ISI)
	return np.sum((subject_rt_histogram-simulated_rt_histogram)**2)

def kernel_merit(params,subject_kernel,model,target,distractor,targetmean,distractormean):
	"""
	Compute least square difference between subject and simulation
	decision kernels. Only take into account the first 1000ms of
	the kernel.
	"""
	model.threshold = params[0]
	simulation = model.batchInference(target,distractor)
	selection = 1.-simulation[:,1]
	fluctuations = np.transpose(np.array([target.T-targetmean,distractor.T-distractormean]),(2,0,1))
	sim_kernel,_,_,_ = ke.kernels(fluctuations,selection,np.ones_like(selection))
	return np.sum((subject_kernel-sim_kernel[:,:25])**2)

class Objective():
	def __init__(self,weight_rt=0.,weight_rtd=0.,weight_k=0.,weight_p=0.,likelihood=False):
		self._rt = 0.
		self._rtd = 0.
		self._k = 0.
		self._p = 0.
		if likelihood:
			self.likelihood = True
			self.name = 'nloglikelihood'
		else:
			self.likelihood = False
			name = []
			if all([weight_rt==0.,weight_rtd==0.,weight_k==0.,weight_p==0.]):
				raise ValueError("No fit objective set. All weights were equal to 0")
			if weight_rt:
				warnings.warn("Direct fitting of RT values is deprecated and will be removed in the future",DeprecationWarning)
				self._rt = weight_rt
				name.append('rt')
			if weight_rtd:
				self._rtd = weight_rtd
				name.append('rtd')
			if weight_k:
				self._k = weight_k
				name.append('kernel')
			if weight_p:
				self._p = weight_p
				name.append('performance')
			self.name = '-'.join([str(n) for n in list(name)])
	
	def __call__(self,params,subject_rt,subject_rtd,subject_kernel,subject_performance,model,bin_edges,target,distractor,targetmean,distractormean):
		if likelihood:
			lp = len(params)
			model.threshold = params[0]
			dead_time_sigma = 1
			dead_time = 0
			if lp>1:
				dead_time = params[1]
				if lp>2:
					dead_time_sigma = params[2]
			simulation = model.batchInference(target,distractor)
			output = np.sum(((subject_rt-simulation[:,3]-params[1])/dead_time_sigma)**2)
		else:
			output = 0.
			model.threshold = params[0]
			simulation = model.batchInference(target,distractor)
			if self._rt:
				output+= self._rt * np.sum((subject_rt-simulation[:,3]-params[1])**2)
			if self._rtd:
				# Squared difference between rt histograms
				simulated_rtd,_ = np.histogram(simulation[:,3]+params[1],bin_edges,density=True)
				if len(params)>2:
					simulated_rtd = conv_hist(simulated_rtd,params[2],model.ISI)
				output+= self._rtd * np.sum((subject_rtd-simulated_rtd)**2)
			if self._k:
				# Squared difference between decision kernels
				selection = 1.-simulation[:,1]
				fluctuations = np.transpose(np.array([target.T-targetmean,distractor.T-distractormean]),(2,0,1))
				sim_kernel,_,_,_ = ke.kernels(fluctuations,selection,np.ones_like(selection))
				window = min(subject_kernel.shape[1],sim_kernel.shape[1])
				output+= self._k * np.sum((subject_kernel[:,:window]-sim_kernel[:,:window])**2)
			if self._p:
				# Pearson chi squared statistic to test whether the simulated and subject performances come from the same distribution
				hits = np.array([np.sum(subject_performance),np.nansum(simulation[:,1])])
				misses = np.array([len(subject_performance),np.nansum(np.logical_not(np.isnan(simulation[:,1])))])-hits
				contingency = np.array([hits,misses])
				expected = np.dot(np.sum(contingency,axis=1,keepdims=True),np.sum(contingency,axis=0,keepdims=True))/np.sum(contingency)
				output+= self._p*np.nansum((contingency-expected)**2/expected)
		return output
	
	def __getstate__(self):
		return {'name':self.name,'likelihood':self.likelihood,'_rt':self._rt,'_rtd':self._rtd,'_k':self._k,'_p':self._p}
	
	def __setstate__(self,state):
		self.name = state['name']
		self.likelihood = state['likelihood']
		self._rt = state['_rt']
		self._rtd = state['_rtd']
		self._k = state['_k']
		self._p = state['_p']

def load_fitter(filename):
	if not filename.endswith('.pkl'):
		filename+='.pkl'
	f = open(filename,'r')
	output = pickle.load(f)
	f.close()
	return output

def resave(directory='fits/'):
	"""
	Used to update the Fitter instances saved in the specified directory
	to use the Objective class instead of the _merit functions
	"""
	if not directory.endswith('/'):
		directory+='/'
	files = sorted([f for f in os.listdir(directory) if f.endswith(".pkl")])
	for f in files:
		fitter = load_fitter(directory+f)
		fitter.save(directory+f,overwrite=True)
		fitter2 = load_fitter(directory+f)
		print fitter.__getstate__()
		print fitter2.__getstate__()

class Fitter():
	def __init__(self,subject,model=None,objective='rt distribution',initial_parameters=None,ubounds=None,\
				criteria=None,synthetic_trials=None,cma_sigma=1/3,cma_options=cma.CMAOptions(),use_subject_signals=False):
		# Subject data
		self.subject = subject
		# Model
		if model:
			self.model = model
		else:
			self.model = pe.UnknownVarPerfectInference()
			self.model.ISI = 40.
		if criteria is not None:
			if criteria.lower()=='optimal':
				self.model.criteria = optimal_criteria
			elif criteria.lower()=='dprime':
				self.model.criteria = dprime_criteria
			elif criteria.lower()=='dprime-var':
				self.model.criteria = dprime_var_criteria
			elif criteria.lower()=='var':
				self.model.criteria = var_criteria
			else:
				raise(ValueError("Unknown criteria %s",criteria))
		else:
			# Use the supplied model's criteria
			pass
		# Objective function (merit)
		if isinstance(objective,Objective):
			self.objective = objective
			self.use_subject_signals = use_subject_signals
			if initial_parameters is not None:
				self.initial_parameters = initial_parameters
			else:
				self.initial_parameters = np.random.random(2)
			if objective.name=='rtd':
				self.fittype = 0
			elif objective.name=='rt':
				self.fittype = 1
				self.use_subject_signals = True
				self.initial_parameters = self.initial_parameters[:2]
			elif objective.name=='kernel':
				self.fittype = 2
				self.initial_parameters = self.initial_parameters[:1]
			else:
				self.fittype = 3
			if self.use_subject_signals and (synthetic_trials is not None):
				warnings.warn("Variable synthetic_trials is not used when use_subject_signals is True",RuntimeWarning)
				self.synthetic_trials = None
			else:
				self.synthetic_trials = synthetic_trials
		else:
			if objective.lower() in ('rtd','rt distribution','rt histogram','rt_distribution','rt_histogram'):
				self.objective = Objective(weight_rtd=1.)
				self.use_subject_signals = use_subject_signals
				self.fittype = 0
				if initial_parameters is not None:
					self.initial_parameters = initial_parameters
				else:
					self.initial_parameters = np.random.random(2)
				if self.use_subject_signals and (synthetic_trials is not None):
					warnings.warn("Variable synthetic_trials is not used when use_subject_signals is True",RuntimeWarning)
					self.synthetic_trials = None
				else:
					self.synthetic_trials = synthetic_trials
			elif objective.lower()=='rt':
				self.objective = Objective(weight_rt=1.)
				if not use_subject_signals:
					warnings.warn("When the objective is to fit subject RT values (not the distribution), use_subject_values is always set as True",RuntimeWarning)
				self.use_subject_signals = True
				if len(initial_parameters)!=2:
					raise(ValueError("If the objective is to fit the RT values, only the model threshold and dead time can be fitted, not the dead time dispersion. Thus, initial_parameters must have only two elements"))
				self.fittype = 1
				self.synthetic_trials = None
				if synthetic_trials is not None:
					warnings.warn("Variable synthetic_trials is not used when the objective is to fit the RT values",RuntimeWarning)
				self.initial_parameters = initial_parameters
			elif objective.lower()=='kernel':
				self.objective = Objective(weight_k=1.)
				self.use_subject_signals = use_subject_signals
				self.fittype = 2
				if initial_parameters is not None:
					self.initial_parameters = initial_parameters[:1]
				else:
					self.initial_parameters = np.random.random(1)
				if self.use_subject_signals and (synthetic_trials is not None):
					warnings.warn("Variable synthetic_trials is not used when use_subject_signals is True",RuntimeWarning)
					self.synthetic_trials = None
				else:
					self.synthetic_trials = synthetic_trials
			else:
				raise(ValueError("Unknown fit objective: %s",objective))
		# CMA parameters
		self.cma_sigma = cma_sigma
		self.cma_options = cma_options
		if ubounds is not None:
			if len(ubounds)!=len(initial_parameters):
				raise(ValueError("Parameter upper boundary must be an array with the same length as the supplied initial_parameters"))
			self.cma_options.set('bounds',[np.zeros(len(self.initial_parameters)),ubounds[:len(self.initial_parameters)]])
			self.cma_options.set('scaling_of_variables',ubounds[:len(self.initial_parameters)])
		
		# Fitting output
		self.synthetic_data = {'target':None,'distractor':None,'indeces':None}
		self.fit_output = (None,None,None,None,None,None,None,None)
	
	def fit(self,restarts=0,**kwargs):
		# Load subject data
		dat,t,d = self.subject.load_data()
		t = np.mean(t,axis=2)
		d = np.mean(d,axis=2)
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
			args = (None,subject_rt_histogram,None,None,self.model,bin_edges,target,distractor,None,None)
		elif self.fittype==1:
			args = (subject_rt,None,None,None,self.model,None,target,distractor,None,None)
		elif self.fittype==2:
			fluctuations = np.transpose(np.array([t.T-dat[:,0],d.T-50]),(2,0,1))
			subject_kernel,_,_,_ = ke.kernels(fluctuations,1-dat[:,2],dat[:,3]-1)
			args = (None,None,subject_kernel,None,self.model,None,target,distractor,dat[indeces,0],50)
		else:
			subject_rt_histogram,_ = np.histogram(subject_rt,bin_edges,density=True)
			fluctuations = np.transpose(np.array([t.T-dat[:,0],d.T-50]),(2,0,1))
			subject_kernel,_,_,_ = ke.kernels(fluctuations,1-dat[:,2],dat[:,3]-1)
			args = (subject_rt,subject_rt_histogram,subject_kernel,dat[:,1],self.model,bin_edges,target,distractor,dat[indeces,0],50)
		cma_fmin_output = cma.fmin(self.objective,self.initial_parameters,self.cma_sigma,options=self.cma_options,args=args,restarts=restarts,**kwargs)
		stop_cond = {}
		for key in cma_fmin_output[7].keys():
			stop_cond[key] = cma_fmin_output[7][key]
		self.fit_output = {'xopt':cma_fmin_output[0],'fopt':cma_fmin_output[1],'evalsopt':cma_fmin_output[2],\
						   'evals':cma_fmin_output[3],'iterations':cma_fmin_output[4],'xmean':cma_fmin_output[5],\
						   'stds':cma_fmin_output[6],'stop':stop_cond}
		return self.fit_output
	
	def __setstate__(self,state):
		self.subject = state['subject']
		self.model = state['model']
		self.use_subject_signals = state['use_subject_signals']
		self.fittype = state['fittype']
		self.synthetic_trials = state['synthetic_trials']
		self.initial_parameters = state['initial_parameters']
		if 'merit' in state.keys() and 'objective' not in state.keys():
			state['objective'] = state['merit']
		if isinstance(state['objective'],Objective):
			self.objective = state['objective']
		else:
			if state['objective'].__name__=="rt_merit":
				self.objective = Objective(weight_rt=1.)
			elif state['objective'].__name__=="rt_histogram_merit":
				self.objective = Objective(weight_rtd=1.)
			elif state['objective'].__name__=="rt_histogram_merit_variable_deadtime":
				self.objective = Objective(weight_rtd=1.)
			elif state['objective'].__name__=="kernel_merit":
				self.objective = Objective(weight_k=1.)
		self.cma_sigma = state['cma_sigma']
		self.cma_options = state['cma_options']
		self.synthetic_data = state['synthetic_data']
		self.fit_output = state['fit_output']
	
	def __getstate__(self):
		return {'subject':self.subject,'model':self.model,'use_subject_signals':self.use_subject_signals,\
				'fittype':self.fittype,'synthetic_trials':self.synthetic_trials,'initial_parameters':self.initial_parameters,\
				'objective':self.objective,'cma_sigma':self.cma_sigma,'cma_options':self.cma_options,\
				'synthetic_data':self.synthetic_data,'fit_output':self.fit_output}
	
	def get_criteria(self):
		cri_name = self.model.criteria.func_name
		if cri_name not in ('criteria','optimal_criteria','dprime_criteria','var_criteria','dprime_var_criteria'):
			criteria = cri_name
		else:
			if cri_name in ('criteria','optimal_criteria'):
				criteria = 'optimal'
			elif cri_name=='dprime_criteria':
				criteria = 'dprime'
			elif cri_name=='dprime_var_criteria':
				criteria = 'dprime-var'
			elif cri_name=='var_criteria':
				criteria = 'var'
		return criteria
	
	def get_objective(self):
		if self.fittype==0:
			objective = 'rtd'
		elif self.fittype==1:
			objective = 'rt'
		elif self.fittype==2:
			objective = 'kernel'
		else:
			objective = self.objective
		return objective
	
	def save(self,filename,overwrite=False):
		if not filename.endswith('.pkl'):
			filename+='.pkl'
		if os.path.isfile(filename) and not overwrite:
			raise(IOError("Supplied file %s already exits. If use wish to overwrite it, call save(...,overwrite=True) instead",filename))
		f = open(filename,'wb')
		pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
		f.close()

def fit_all_subjects(data_dir='/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles',subject_id='all',criteria='all',objective='all'):
	# Fit parameters for each subject and then for all subjects
	subjects = io.unique_subjects(data_dir)
	subjects.append(io.merge_subjects(subjects))
	criterias = ['optimal','dprime','dprime-var','var']
	objectives = ['rt','rtd','kernel']
	if subject_id!='all':
		subjects = [s for s in itertools.ifilter(lambda s: s.id==subject_id,subjects)]
	if criteria!='all':
		criterias = [c for c in itertools.ifilter(lambda c: c==criteria, criterias)]
	if objective!='all':
		if isinstance(objective,tuple):
			objectives = [Objective(*objective)]
		else:
			objectives = [o for o in itertools.ifilter(lambda c: c==objective, objectives)]
	ISI = 40.
	priors={'prior_mu_t':50,'prior_mu_d':50,'prior_va_t':15**2,'prior_va_d':15**2}
	model = pe.KnownVarPerfectInference(model_var_t=25.,model_var_d=25.,ISI=ISI,**priors)
	initial_parameters = [1.9678714549600138, 621.71742870406183, 75.978678076060476]
	ubounds = [20.,5000.,200.]
	counter = 0
	for s in subjects:
		for criteria in criterias:
			for objective in objectives:
				counter+=1
				print counter
				if objective=='rt':
					init_pars = initial_parameters[:2]
					ub = ubounds[:2]
				else:
					init_pars = initial_parameters
					ub = ubounds
				fitter = Fitter(s,model,objective=objective,initial_parameters=init_pars,ubounds=ub,\
						criteria=criteria,synthetic_trials=10000,cma_sigma=1/3,cma_options=cma.CMAOptions(),use_subject_signals=False)
				fitter.fit(restarts=0)
				if isinstance(objective,Objective):
					obj_name = objective.name
				else:
					obj_name = objective
				filename = 'fits/Fit_subject_'+str(s.id)+'_criteria_'+criteria+'_objective_'+obj_name
				fitter.save(filename,overwrite=True)

def analyze_all_fits(fit_dir='fits/',subject_id='all',criteria='all',objective='all',save=False,savefname='fits',group_by=None):
	if not fit_dir.endswith('/'):
		fit_dir+='/'
	files = sorted([f for f in os.listdir(fit_dir) if (f.endswith(".pkl") and f.startswith('Fit_'))])
	figures = []
	ids = []
	counter = 0
	for f in files:
		temp = f.split("_")
		sid = temp[2]
		cri = temp[4]
		obj = temp[6][:-4]
		if sid not in subject_id:
			if subject_id!='all':
				continue
		sname = 'all' if sid=='0' else str(sid)
		if cri not in criteria:
			if criteria!='all':
				continue
		if obj!=objective:
			if objective!='all':
				continue
		counter+=1
		
		fitter = load_fitter(fit_dir+f)
		subject = fitter.subject
		model = fitter.model
		target = fitter
		# Load subject data and compute RT distribution and kernels
		dat,t,d = subject.load_data()
		max_rt_ind = int(math.ceil(max(dat[:,1])/model.ISI))
		bin_edges = np.array(range(max_rt_ind+1))*model.ISI
		subject_rt,_ = np.histogram(dat[:,1],bin_edges,density=True)
		t = (np.mean(t,axis=2,keepdims=False).T-dat[:,0]).T
		d = np.mean(d,axis=2,keepdims=False)-50
		fluctuations = np.transpose(np.array([t,d]),(1,0,2))
		sdk,sck,sdk_std,sck_std = ke.kernels(fluctuations,1-dat[:,2],dat[:,3]-1)
		sT = np.array(range(sdk.shape[1]),dtype=float)*model.ISI
		
		target = fitter.synthetic_data['target']
		distractor = fitter.synthetic_data['distractor']
		indeces = fitter.synthetic_data['indeces']
		model.threshold = fitter.fit_output['xopt'][0]
		if len(fitter.fit_output['xopt'])>1:
			if fitter.fittype!=2:
				dead_time = fitter.fit_output['xopt'][1]
			else:
				dead_time = 0.
		else:
			dead_time = 0.
		if len(fitter.fit_output['xopt'])>2:
			if fitter.fittype!=1 and fitter.fittype!=2:
				var_dead_time = fitter.fit_output['xopt'][2]
			else:
				var_dead_time = None
		else:
			var_dead_time = None
		simulation = model.batchInference(target,distractor)
		simulated_rt,_ = np.histogram(simulation[:,3]+dead_time,bin_edges,density=True)
		if var_dead_time:
			simulated_rt = conv_hist(simulated_rt,var_dead_time,model.ISI)
		simulated_sel = 1-simulation[:,1]
		simulated_sel[np.isnan(simulated_sel)] = 1
		sim_fluctuations = np.transpose(np.array([target.T-dat[indeces,0],distractor.T-50]),(2,0,1))
		sim_sdk,sim_sck,sim_sdk_std,sim_sck_std = ke.kernels(sim_fluctuations,simulated_sel,np.ones_like(simulated_sel))
		sim_sT = np.array(range(sim_sdk.shape[1]),dtype=float)*model.ISI
		sim_sT[sdk.shape[1]:] = np.nan
		sim_sdk[:,sdk.shape[1]:] = np.nan
		sim_sck[:,sdk.shape[1]:] = np.nan
		sim_sdk_std[:,sdk.shape[1]:] = np.nan
		sim_sck_std[:,sdk.shape[1]:] = np.nan
		
		if not interactive_pyplot:
			plt.ioff() # cma overrides original pyplot settings and if interactive mode was off, the plot is never displayed!
		
		suptitle = []
		label = ['Sim']
		if group_by is not None:
			if group_by=='subject':
				suptitle.append('subject='+sname)
			else:
				label.append('s='+sname)
			if group_by=='criteria':
				suptitle.append('criteria='+cri)
			else:
				label.append('c='+cri)
			if group_by=='objective':
				suptitle.append('objective='+obj)
			else:
				label.append('o='+obj)
			suptitle = '; '.join(suptitle).capitalize()
			label = '; '.join(label)
		else:
			suptitle = 'Subject='+sname+'; criteria='+cri+'; objective='+obj
			label = 'Simulation'
		fig = plt.figure(suptitle,figsize=(13,10))
		plot_subject_data = True
		if fig not in figures:
			figures.append(fig)
			ids.append(sid)
		elif group_by=='subject':
			plot_subject_data = False
		plt.subplot(121)
		if plot_subject_data:
			plt.plot(bin_edges[:-1],subject_rt,'k',label='Subject',linewidth=2)
		plt.plot(bin_edges[:-1],simulated_rt,label=label)
		plt.legend()
		plt.xlabel('Response time [ms]')
		plt.ylabel('Prob density [1/ms]')
		plt.subplot(122)
		if group_by is None:
			plt.plot(sT,sdk[0],color='b',linestyle='--',label='Subject $D_{S}$')
			plt.plot(sT,sdk[1],color='r',linestyle='--',label='Subject $D_{N}$')
			plt.fill_between(sT,sdk[0]-sdk_std[0],sdk[0]+sdk_std[0],color='b',alpha=0.3,edgecolor=None)
			plt.fill_between(sT,sdk[1]-sdk_std[1],sdk[1]+sdk_std[1],color='r',alpha=0.3,edgecolor=None)
			plt.plot(sim_sT,sim_sdk[0],color='b',label='Simulation $D_{S}$')
			plt.plot(sim_sT,sim_sdk[1],color='r',label='Simulation $D_{N}$')
			plt.fill_between(sim_sT,sim_sdk[0]-sim_sdk_std[0],sim_sdk[0]+sim_sdk_std[0],color='b',alpha=0.3,edgecolor=None)
			plt.fill_between(sim_sT,sim_sdk[1]-sim_sdk_std[1],sim_sdk[1]+sim_sdk_std[1],color='r',alpha=0.3,edgecolor=None)
		else:
			if plot_subject_data:
				line1 = plt.plot(sT,sdk[0],color='k',linestyle='--',label='Subject')[0]
				plt.plot(sT,sdk[1],color=line1.get_color(),linestyle='--')
				plt.fill_between(sT,sdk[0]-sdk_std[0],sdk[0]+sdk_std[0],color=line1.get_color(),alpha=0.3,edgecolor=None)
				plt.fill_between(sT,sdk[1]-sdk_std[1],sdk[1]+sdk_std[1],color=line1.get_color(),alpha=0.3,edgecolor=None)
			line1 = plt.plot(sim_sT,sim_sdk[0],label=label)[0]
			plt.plot(sim_sT,sim_sdk[1],color=line1.get_color())
			plt.fill_between(sim_sT,sim_sdk[0]-sim_sdk_std[0],sim_sdk[0]+sim_sdk_std[0],color=line1.get_color(),alpha=0.3,edgecolor=None)
			plt.fill_between(sim_sT,sim_sdk[1]-sim_sdk_std[1],sim_sdk[1]+sim_sdk_std[1],color=line1.get_color(),alpha=0.3,edgecolor=None)
		plt.plot(plt.gca().get_xlim(),[0,0],color='k')
		plt.xlabel('Time [ms]')
		plt.ylabel('Fluctuation [$cd/m^{2}$]')
		plt.legend()
		plt.suptitle(suptitle)
	if save:
		from matplotlib.backends.backend_pdf import PdfPages
		with PdfPages('../../figs/'+savefname+'.pdf') as pdf:
			for fig in [f for (i,f) in sorted(zip(ids,figures), key=lambda pair: pair[0])]:
				pdf.savefig(fig)
	else:
		plt.show()
	if counter==0:
		print "No fit exists with the desired subject_id, criteria and objective"

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
	
	fitter = Fitter(ms,model,objective='rt distribution',initial_parameters=initial_parameters,ubounds=ubounds,\
				criteria=None,synthetic_trials=10000,cma_sigma=1/3,cma_options=cma.CMAOptions(),use_subject_signals=False)
	
	fit_output = fitter.fit(restarts=0)
	fitter.save('Fit_test',overwrite=True)
	fitter = load_fitter('Fit_test')
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

def parse_args():
	""" Parse arguments passed from the command line """
	arg_list = ['program','dir','subject','criteria','objective','save','group_by','savefname']
	options = {'program':'test','dir':None,'subject':'all','criteria':'all','objective':'all','save':False,'group_by':None,'savefname':'fits'}
	kwarg_flag = False
	skip_arg = True
	mixed_objective = False
	arg_n = 0
	for c,arg in enumerate(sys.argv):
		if skip_arg:
			skip_arg = False
			continue
		if not kwarg_flag and arg.startswith('-'):
			kwarg_flag = True
		elif not kwarg_flag:
			if c==1:
				if arg.lower() not in ['test','fit','analyze']:
					raise ValueError("Supplied program must be either 'test', 'fit' or 'analyze'. User supplied '%s' instead" % (arg))
				arg = arg.lower()
			elif c==2:
				pass
			elif c==3:
				if arg.lower()!='all':
					try:
						int(arg)
					except ValueError:
						raise ValueError("Supplied subject must be either 'all' or a subject id. User supplied '%s' instead" % (arg))
				arg = arg.lower()
			elif c==4:
				if arg.lower() not in ['all','optimal','dprime','dprime-var','var']:
					raise ValueError("Supplied criteria must be either 'all', 'optimal', 'dprime', 'dprime-var' or 'var'. User supplied '%s' instead" % (arg))
				arg = arg.lower()
			elif c==5:
				if arg.startswith('(') and arg.endswith(')'): # Passing a tuple of values to construct a mixed Objective
					try:
						parsed_arg = arg[1:-1] # Trim the parenthesis from the tuple
						parsed_arg.split(',')
						parsed_arg = tuple([float(a) for a in parsed_arg])
						arg = parsed_arg
					except:
						print "Could not properly parse the tuple supplied to construct Objective. Supplied tuple: %s"%(arg)
						raise
				else:
					arg = arg.lower()
					if arg not in ['all','rt','rtd','kernel']:
						if '-' in arg:
							if not all([(a in ['rt','rtd','kernel']) for a in arg.split('-')]):
								raise ValueError("Supplied objective must be either 'all', 'rt', 'rtd' or 'kernel'. User supplied '%s' instead" % (arg))
							else:
								mixed_objective = True
					arg = arg.lower()
			elif c==6:
				if arg.lower() not in ['true','false']:
					raise ValueError("Save must be either 'true' or 'false'. User supplied '%s' instead" % (arg))
				arg = arg=='true'
			elif c==7:
				if arg.lower() not in ['none','subject','criteria','objective']:
					raise ValueError("Group_by must be either 'none', 'subject', 'criteria' or 'objective'. User supplied '%s' instead" % (arg))
				arg = arg.lower()
				arg = arg if arg!='none' else None
			elif c==8:
				pass
			elif c>8:
				raise Exception("Unknown sixth option supplied '%s'" %s)
			options[arg_list[arg_n]] = arg
			arg_n+=1
		if kwarg_flag:
			skip_arg = True
			if not arg.startswith('-'):
				raise ValueError("Expected -key value pair options and found a positional argument instead")
			key = arg[1:].lower()
			if key in ['program','subject','criteria','objective']:
				val = sys.argv[c+1].lower()
				if key=='program':
					if val not in ['test','fit','analyze']:
						raise ValueError("Supplied program must be either 'test', 'fit' or 'analyze'. User supplied '%s' instead" % (val))
				elif key=='subject':
					if arg!='all':
						try:
							int(val)
						except ValueError:
							raise ValueError("Supplied subject must be either 'all' or a subject id. User supplied '%s' instead" % (val))
				elif key=='criteria':
					if val not in ['all','optimal','dprime','dprime-var','var']:
						raise ValueError("Supplied criteria must be either 'all', 'optimal', 'dprime', 'dprime-var' or 'var'. User supplied '%s' instead" % (val))
				elif key=='objective':
					if val.startswith('(') and val.endswith(')'): # Passing a tuple of values to construct a mixed Objective
						try:
							parsed_val = val[1:-1] # Trim the parenthesis from the tuple
							parsed_val = parsed_val.split(',')
							parsed_val = tuple([float(a) for a in parsed_val])
							val = parsed_val
						except:
							print "Could not properly parse the tuple supplied to construct Objective. Supplied tuple: %s"%(val)
							raise
					else:
						if val.lower() not in ['all','rt','rtd','kernel']:
							if '-' in val:
								if not all([(v in ['rt','rtd','kernel']) for v in val.lower().split('-')]):
									raise ValueError("Supplied objective must be either 'all', 'rt', 'rtd' or 'kernel'. User supplied '%s' instead" % (val))
								else:
									mixed_objective = True
						val = val.lower()
			elif key in ['dir','savefname']:
				val = sys.argv[c+1]
			elif key=='save':
				val = sys.argv[c+1].lower()
				if val not in ['true','false']:
					raise ValueError("Save must be either 'true' or 'false'. User supplied '%s' instead" % (val))
				val = val=='true'
			elif key=='group_by':
				val = sys.argv[c+1].lower()
				if val.lower() not in ['none','subject','criteria','objective']:
					raise ValueError("Group_by must be either 'none', 'subject', 'criteria' or 'objective'. User supplied '%s' instead" % (val))
				val = val.lower()
				val = val if val!='none' else None
			elif key in ['h','-help']:
				display_help()
			else:
				raise Exception("Unknown option '%s' supplied" % (key))
			options[key] = val
	if mixed_objective and (not options['program']=='analyze'):
		raise ValueError("Mixed objective name is only accepted in 'analyze' program. For other programs use the tuple of weigths.")
	print options
	return options

def display_help():
	h = """ order_effects.py help
 Sintax:
 behavioral_fit.py [optional arguments]
 
 behavioral_fit.py -h [or --help] displays help
 
 Optional arguments are:
 'program': Can be 'test', 'fit' or 'analyze'. Determines whether to run the test suite, run the fit_all_subjects function or analyze_all_fits function. [default 'test']
 'dir': The directory from which to fetch the data. If the program is test or fit, it must be the path to the behavioral data files. If program is analyze, dir must be the path to the fit files. [default 'None' and fall to each function's default]
 'subject': The subject you with to fit or analyze. Can be 'all' saying you with to analyze all subjects (including the merge of all subjects that has id=0) or an integer id 0(represents the merge of all subjects), 1, 2, etc. [default 'all']
 'criteria': The criteria to implement. Posible values are 'optimal', 'dprime', 'var' and 'dprime-var'. [default 'optimal']
 'objective': The objective function to fit or analyze. Can be 'all', 'rt', 'rtd' or 'kernel'. If using the 'fit' program, one can supply a tuple of four elements to construct the custom Objective instance (weight_rt,weight_rtd,weight_k,weight_p). [default 'all']
 'save': Only used with the analyze program. Can be 'false' or 'true' and determines whether the graphs are shown or saved to ../../figs/fits.pdf respectively. [default 'false']
 'group_by': Only used with the analyze program. Can be 'none', 'subject', 'criteria' or 'objective' and determines whether plots are grouped by a given property. [default 'none']
 'savefname': Only used with the analyze program if save is True. Is the filename to which to save the plots. [default: 'fits']
 
 Argument can be supplied as positional arguments or as -key value pairs.
 
 Example:
 python behavioral_fit.py fit -subject 0 """
	print h
	exit()


if __name__=="__main__":
	args = parse_args()
	if args['program']=='test':
		if args['dir'] is not None:
			test(args['dir'])
		else:
			test()
	elif args['program']=='fit':
		if args['dir'] is not None:
			fit_all_subjects(data_dir=args['dir'],subject_id=args['subject'],criteria=args['criteria'],objective=args['objective'])
		else:
			fit_all_subjects(subject_id=args['subject'],criteria=args['criteria'],objective=args['objective'])
	elif args['program']=='analyze':
		if args['dir'] is not None:
			analyze_all_fits(fit_dir=args['dir'],subject_id=args['subject'],criteria=args['criteria'],objective=args['objective'],group_by=args['group_by'],savefname=args['savefname'])
		else:
			analyze_all_fits(subject_id=args['subject'],criteria=args['criteria'],objective=args['objective'],save=args['save'],group_by=args['group_by'],savefname=args['savefname'])
