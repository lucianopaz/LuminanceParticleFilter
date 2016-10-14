from __future__ import division
from __future__ import print_function

import enum, os, sys, math, scipy, pickle, warnings, json, logging, copy, re
import scipy.signal
import numpy as np
from utils import normpdf,average_downsample

class Location(enum.Enum):
	facu = 0
	home = 1
	cluster = 2
	unknown = 3

opsys,computer_name,kern,bla,bits = os.uname()
if opsys.lower().startswith("linux"):
	if computer_name=="facultad":
		loc = Location.facu
	elif computer_name.startswith("sge") or computer_name.startswith("slurm"):
		loc = Location.cluster
elif opsys.lower().startswith("darwin"):
	loc = Location.home
else:
	loc = Location.unknown

if loc==Location.facu:
	raw_data_dir='/home/luciano/Dropbox/Luciano/datos joaquin/para_luciano/raw_data'
elif loc==Location.home:
	raw_data_dir='/Users/luciano/Dropbox/Luciano/datos joaquin/para_luciano/raw_data'
elif loc==Location.cluster:
	raw_data_dir='/homedtic/lpaz/inference/raw_data/raw_data'
elif loc==Location.unknown:
	raise ValueError("Unknown data_dir location")

try:
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		import matplotlib as mt
	if loc==Location.cluster:
		mt.use('Agg')
	from matplotlib import pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	import matplotlib.gridspec as gridspec
	from matplotlib.colors import LogNorm
	can_plot = True
except:
	can_plot = False
import data_io_cognition as io
import cost_time as ct
import cma

def rt_confidence_likelihood(confidence_matrix_time,confidence_matrix,RT,confidence):
	if RT>confidence_matrix_time[-1] or RT<confidence_matrix_time[0]:
		return 0.
	if confidence>1 or confidence<0:
		return 0.
	nC,nT = confidence_matrix.shape
	confidence_array = np.linspace(0,1,nC)
	t_ind = np.interp(RT,confidence_matrix_time,np.arange(0,nT,dtype=np.float))
	c_ind = np.interp(confidence,confidence_array,np.arange(0,nC,dtype=np.float))
	
	floor_t_ind = int(np.floor(t_ind))
	ceil_t_ind = int(np.ceil(t_ind))
	t_weight = 1.-t_ind%1.
	if floor_t_ind==nT-1:
		ceil_t_ind = floor_t_ind
		t_weight = np.array([1.])
	else:
		t_weight = np.array([1.-t_ind%1.,t_ind%1.])
	
	floor_c_ind = int(np.floor(c_ind))
	ceil_c_ind = int(np.ceil(c_ind))
	if floor_c_ind==nC-1:
		ceil_c_ind = floor_c_ind
		c_weight = np.array([1.])
	else:
		c_weight = np.array([1.-c_ind%1.,c_ind%1.])
	weight = np.ones((len(c_weight),len(t_weight)))
	for index,cw in enumerate(c_weight):
		weight[index,:]*= cw
	for index,tw in enumerate(t_weight):
		weight[:,index]*= tw
	
	prob = np.sum(confidence_matrix[floor_c_ind:ceil_c_ind+1,floor_t_ind:ceil_t_ind+1]*weight)
	return prob

def rt_likelihood(t,decision_pdf,RT):
	if RT>t[-1] or RT<t[0]:
		return 0.
	nT = decision_pdf.shape[0]
	t_ind = np.interp(RT,t,np.arange(0,nT,dtype=np.float))
	
	floor_t_ind = int(np.floor(t_ind))
	ceil_t_ind = int(np.ceil(t_ind))
	t_weight = 1.-t_ind%1.
	if floor_t_ind==nT-1:
		ceil_t_ind = floor_t_ind
		weight = np.array([1.])
	else:
		weight = np.array([1.-t_ind%1.,t_ind%1.])
	
	prob = np.sum(decision_pdf[floor_t_ind:ceil_t_ind+1]*weight)
	return prob

def confidence_likelihood(confidence_pdf,confidence):
	if confidence>1. or confidence<0.:
		return 0.
	nC = confidence_pdf.shape[0]
	confidence_array = np.linspace(0.,1.,nC)
	c_ind = np.interp(confidence,confidence_array,np.arange(0,nC,dtype=np.float))
	
	floor_c_ind = int(np.floor(c_ind))
	ceil_c_ind = int(np.ceil(c_ind))
	if floor_c_ind==nC-1:
		ceil_c_ind = floor_c_ind
		weight = np.array([1.])
	else:
		weight = np.array([1.-c_ind%1.,c_ind%1.])
	
	prob = np.sum(confidence_pdf[floor_c_ind:ceil_c_ind+1]*weight)
	return prob

def load_Fitter_from_file(fname):
	f = open(fname,'r')
	fitter = pickle.load(f)
	f.close()
	return fitter

def Fitter_filename(experiment,method,name,session,optimizer,suffix,confidence_map_method='log_odds'):
	if confidence_map_method=='log_odds':
		return 'fits_cognition/{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}{suffix}.pkl'.format(
				experiment=experiment,method=method,name=name,session=session,optimizer=optimizer,suffix=suffix)
	else:
		return 'fits_cognition/{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}_cmapmeth_{cmapmeth}{suffix}.pkl'.format(
				experiment=experiment,method=method,name=name,session=session,optimizer=optimizer,suffix=suffix,cmapmeth=confidence_map_method)

class Fitter:
	#~ __module__ = os.path.splitext(os.path.basename(__file__))[0]
	# Initer
	def __init__(self,subjectSession,time_units='seconds',method='full',optimizer='cma',\
				decisionPolicyKwArgs={},suffix='',rt_cutoff=None,confidence_partition=100,\
				high_confidence_mapping_method='log_odds'):
		logging.debug('Creating Fitter instance for "{experiment}" experiment and "{name}" subject with sessions={session}'.format(
						experiment=subjectSession.experiment,name=subjectSession.name,session=subjectSession.session))
		self.raw_data_dir = raw_data_dir
		self.set_experiment(subjectSession.experiment)
		self.set_time_units(str(time_units))
		self.set_rt_cutoff(rt_cutoff)
		self.set_subjectSession_data(subjectSession)
		self.method = str(method)
		logging.debug('Setted Fitter method = %s',self.method)
		self.optimizer = str(optimizer)
		logging.debug('Setted Fitter optimizer = %s',self.optimizer)
		self.suffix = str(suffix)
		logging.debug('Setted Fitter suffix = %s',self.suffix)
		self.set_decisionPolicyKwArgs(decisionPolicyKwArgs)
		self.dp = ct.DecisionPolicy(**self.decisionPolicyKwArgs)
		self.confidence_partition = int(confidence_partition)
		self.high_confidence_mapping_method = str(high_confidence_mapping_method).lower()
		self.__fit_internals__ = None
	
	# Setters
	def set_experiment(self,experiment):
		self.experiment = str(experiment)
		if self.experiment=='Luminancia':
			self._distractor = 50.
		elif self.experiment=='Auditivo':
			self._distractor = 0.
		elif self.experiment=='2AFC':
			self._distractor = 0.
		else:
			raise ValueError('Fitter class does not support the experiment "{0}"'.format(experiment))
		logging.debug('Setted Fitter experiment = %s',self.experiment)
	
	def set_time_units(self,time_units='seconds'):
		if time_units not in ('seconds','milliseconds'):
			raise ValueError("Invalid time units '{0}'. Available units are seconds and milliseconds".format(units))
		self.time_units = time_units
		logging.debug('Setted Fitter instance time_units = %s',self.time_units)
		
		self._tp = 0.
		logging.debug('Setted Fitter instance _tp = %f',self._tp)
		if self.experiment=='Luminancia':
			logging.debug('Luminancia experiment condition')
			if time_units=='seconds':
				self._ISI = 0.05
				self._T = 1.
				self._iti = 1.5
				self._dt = 0.008
			else:
				self._ISI = 50.
				self._T = 1000.
				self._iti = 1500.
				self._dt = 8.
			#~ self._dt = self._ISI/50.
			self._forced_non_decision_time = 0.
			self._model_var = 50./self._ISI
			self._internal_var = self._model_var
			self._fixed_stim_duration = 0.
			logging.debug('Setted _ISI = %(_ISI)f\t_T = %(_T)f\t_iti = %(_iti)f\t_model_var = %(_model_var)f\t_internal_var = %(_internal_var)f\tfixed_stim_duration = %(_fixed_stim_duration)f',
						  {'_ISI':self._ISI,'_T':self._T,'_iti':self._iti,'_model_var':self._model_var,'_internal_var':self._internal_var,'_fixed_stim_duration':self._fixed_stim_duration})
		elif self.experiment=='Auditivo':
			logging.debug('Auditivo experiment condition')
			if time_units=='seconds':
				self._ISI = 0.5
				self._dt = 0.005
				self._T = 0.3
				self._iti = 2.
				self._forced_non_decision_time = 0.3
			else:
				self._ISI = 500.
				self._dt = 5.
				self._T = 300.
				self._iti = 2000.
				self._forced_non_decision_time = 300.
			self._model_var = 1250.
			self._internal_var = 1250.
			self._fixed_stim_duration = 0.3
			logging.debug('Setted _ISI = %(_ISI)f\t_T = %(_T)f\t_iti = %(_iti)f\t_model_var = %(_model_var)f\t_internal_var = %(_internal_var)f\tfixed_stim_duration = %(_fixed_stim_duration)f',
						  {'_ISI':self._ISI,'_T':self._T,'_iti':self._iti,'_model_var':self._model_var,'_internal_var':self._internal_var,'_fixed_stim_duration':self._fixed_stim_duration})
		elif self.experiment=='2AFC':
			logging.debug('2AFC experiment condition')
			if time_units=='seconds':
				self._ISI = 0.3
				self._dt = 0.005
				self._T = 0.3
				self._iti = 1.5
			else:
				self._ISI = 300.
				self._dt = 5.
				self._T = 300.
				self._iti = 1500.
			self._forced_non_decision_time = 0.
			self._model_var = 1250.
			self._internal_var = 1250.
			self._fixed_stim_duration = 0.3
			logging.debug('Setted _ISI = %(_ISI)f\t_T = %(_T)f\t_iti = %(_iti)f\t_model_var = %(_model_var)f\t_internal_var = %(_internal_var)f\tfixed_stim_duration = %(_fixed_stim_duration)f',
						  {'_ISI':self._ISI,'_T':self._T,'_iti':self._iti,'_model_var':self._model_var,'_internal_var':self._internal_var,'_fixed_stim_duration':self._fixed_stim_duration})
	
	def set_rt_cutoff(self,rt_cutoff=None):
		if rt_cutoff is None:
			rt_cutoff = 14.
		else:
			rt_cutoff = float(rt_cutoff)
		if self.experiment=='Luminancia':
			rt_cutoff = np.min([1.,rt_cutoff])
		if self.time_units=='milliseconds':
			rt_cutoff*=1e3
		self.rt_cutoff = rt_cutoff
		logging.debug('Setted rt_cutoff = %f',self.rt_cutoff)
	
	def set_subjectSession_data(self,subjectSession):
		self.subjectSession = subjectSession
		self._subjectSession_state = subjectSession.__getstate__()
		logging.debug('Setted Fitter _subjectSession_state')
		logging.debug('experiment:%(experiment)s, name:%(name)s, session:%(session)s, data_dir:%(data_dir)s',self._subjectSession_state)
		dat = subjectSession.load_data(override_raw_data_dir={'original':self.raw_data_dir,'replacement':raw_data_dir})
		logging.debug('Loading subjectSession data')
		self.rt = dat[:,1]
		if self.time_units=='milliseconds':
			self.rt*=1e3
		self.rt+=self._forced_non_decision_time
		valid_trials = self.rt<=self.rt_cutoff
		self.rt = self.rt[valid_trials]
		self.max_RT = np.max(self.rt)
		self.min_RT = np.min(self.rt)
		dat = dat[valid_trials]
		
		trials = len(self.rt)
		if trials==0:
			raise RuntimeError('No trials can be fitted')
		
		self.contrast = dat[:,0]-self._distractor
		if self.experiment=='Luminancia':
			self._contrast = self.contrast/self._ISI
		elif self.experiment=='2AFC':
			self._contrast = self.contrast/self._ISI
		else:
			self._contrast = self.contrast/self._forced_non_decision_time
		self.performance = dat[:,2]
		self.confidence = dat[:,3]
		logging.debug('Trials loaded = %d',len(self.performance))
		self.mu,self.mu_indeces,count = np.unique(self._contrast,return_inverse=True,return_counts=True)
		logging.debug('Number of different drifts = %d',len(self.mu))
		self.mu_prob = count.astype(np.float64)/np.sum(count.astype(np.float64))
		if self.mu[0]==0:
			mus = np.concatenate((-self.mu[-1:0:-1],self.mu))
			p = np.concatenate((self.mu_prob[-1:0:-1],self.mu_prob))
			p[mus!=0]*=0.5
		else:
			mus = np.concatenate((-self.mu[::-1],self.mu))
			p = np.concatenate((self.mu_prob[::-1],self.mu_prob))*0.5
		
		self._prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
		logging.debug('Setted Fitter _prior_mu_var = %f',self._prior_mu_var)
	
	def set_decisionPolicyKwArgs(self,decisionPolicyKwArgs):
		defaults = self.default_decisionPolicyKwArgs()
		defaults.update(decisionPolicyKwArgs)
		self.decisionPolicyKwArgs = defaults
		logging.debug('Setted Fitter decisionPolicyKwArgs = %s',self.decisionPolicyKwArgs)
	
	def __setstate__(self,state):
		self.set_experiment(state['experiment'])
		self.set_time_units(state['time_units'])
		self.set_rt_cutoff(state['rt_cutoff'])
		self.raw_data_dir = state['raw_data_dir']
		if 'confidence_partition' in state.keys():
			self.confidence_partition = int(state['confidence_partition'])
		else:
			self.confidence_partition = 100.
		self.method = state['method']
		self.optimizer = state['optimizer']
		self.suffix = state['suffix']
		self.set_subjectSession_data(io.SubjectSession(name=state['subjectSession_state']['name'],
													   session=state['subjectSession_state']['session'],
													   experiment=self.experiment,
													   data_dir=state['subjectSession_state']['data_dir']))
		self.set_decisionPolicyKwArgs(state['decisionPolicyKwArgs'])
		self.dp = ct.DecisionPolicy(**self.decisionPolicyKwArgs)
		if '_start_point' in state.keys():
			self._start_point = state['_start_point']
		if '_bounds' in state.keys():
			self._bounds = state['_bounds']
		if '_fitted_parameters' in state.keys():
			self._fitted_parameters = state['_fitted_parameters']
		if '_fixed_parameters' in state.keys():
			self._fixed_parameters = state['_fixed_parameters']
		if 'fit_arguments' in state.keys():
			self.fixed_parameters = state['fit_arguments']['fixed_parameters']
			self.fitted_parameters = state['fit_arguments']['fitted_parameters']
			self._fit_arguments = state['fit_arguments']
		if 'fit_output' in state.keys():
			self._fit_output = state['fit_output']
		if 'high_confidence_mapping_method' is state.keys():
			self.high_confidence_mapping_method = state['high_confidence_mapping_method']
		else:
			self.high_confidence_mapping_method = 'log_odds'
		self.__fit_internals__ = None
	
	def set_fixed_parameters(self,fixed_parameters={}):
		defaults = self.default_fixed_parameters()
		defaults.update(fixed_parameters)
		fittable_parameters = self.get_fittable_parameters()
		self._fixed_parameters = fixed_parameters.copy()
		self._fitted_parameters = []
		for par in fittable_parameters:
			if par not in self._fixed_parameters.keys():
				self._fitted_parameters.append(par)
		logging.debug('Setted Fitter fixed_parameters = %s',self._fixed_parameters)
		logging.debug('Setted Fitter fitted_parameters = %s',self._fitted_parameters)
	
	def set_start_point(self,start_point={}):
		defaults = self.default_start_point()
		defaults.update(start_point)
		self._start_point = defaults
		logging.debug('Setted Fitter start_point = %s',self._start_point)
	
	def set_bounds(self,bounds={}):
		defaults = self.default_bounds()
		defaults.update(bounds)
		self._bounds = defaults
		logging.debug('Setted Fitter bounds = %s',self._bounds)
	
	def set_optimizer_kwargs(self,optimizer_kwargs={}):
		defaults = self.default_optimizer_kwargs()
		defaults.update(optimizer_kwargs)
		self.optimizer_kwargs = defaults
		logging.debug('Setted Fitter optimizer_kwargs = %s',self.optimizer_kwargs)
	
	# Getters
	def get_parameters_dict(self):
		parameters = self.get_fixed_parameters().copy()
		try:
			start_point = self.get_start_point()
		except:
			start_point = self.default_start_point()
		for fp in self.get_fitted_parameters():
			parameters[fp] = start_point[fp]
		return parameters
	
	def get_parameters_dict_from_array(self,x):
		parameters = self.fixed_parameters.copy()
		for index,key in enumerate(self.fitted_parameters):
			parameters[key] = x[index]
		return parameters
	
	def get_parameters_dict_from_fit_output(self,fit_output=None):
		if fit_output is None:
			fit_output = self._fit_output
		parameters = self.fixed_parameters.copy()
		parameters.update(fit_output[0])
		return parameters
	
	def get_fixed_parameters(self):
		try:
			return self.fixed_parameters
		except:
			try:
				return self._fixed_parameters
			except:
				return self.default_fixed_parameters()
	
	def get_fitted_parameters(self):
		try:
			return self.fitted_parameters
		except:
			try:
				return self._fitted_parameters
			except:
				return [p for p in self.get_fittable_parameters() if p not in self.default_fixed_parameters().keys()]
	
	def get_start_point(self):
		return self._start_point
	
	def get_bounds(self):
		return self._bounds
	
	def __getstate__(self):
		state = {'experiment':self.experiment,
				 'time_units':self.time_units,
				 'rt_cutoff':self.rt_cutoff,
				 'subjectSession_state':self._subjectSession_state,
				 'method':self.method,
				 'optimizer':self.optimizer,
				 'suffix':self.suffix,
				 'decisionPolicyKwArgs':self.decisionPolicyKwArgs,
				 'confidence_partition':self.confidence_partition,
				 'raw_data_dir':self.raw_data_dir,
				 'high_confidence_mapping_method':self.high_confidence_mapping_method}
		if hasattr(self,'_start_point'):
			state['_start_point'] = self._start_point
		if hasattr(self,'_bounds'):
			state['_bounds'] = self._bounds
		if hasattr(self,'_fitted_parameters'):
			state['_fitted_parameters'] = self._fitted_parameters
		if hasattr(self,'_fixed_parameters'):
			state['_fixed_parameters'] = self._fixed_parameters
		if hasattr(self,'_fit_arguments'):
			state['fit_arguments'] = self._fit_arguments
		if hasattr(self,'_fit_output'):
			state['fit_output'] = self._fit_output
		return state
	
	def get_fittable_parameters(self):
		return self.get_decision_parameters()+self.get_confidence_parameters()
	
	def get_decision_parameters(self):
		return ['cost','dead_time','dead_time_sigma','phase_out_prob','internal_var']
	
	def get_confidence_parameters(self):
		return ['high_confidence_threshold','confidence_map_slope']
	
	def get_dead_time_convolver(self,parameters):
		must_downsample = True
		if self.time_units=='seconds' and self.dp.dt>1e-3:
			_dt = 1e-3
		elif self.time_units=='milliseconds' and self.dp.dt>1.:
			_dt = 1.
		else:
			must_downsample = False
			_dt = self.dp.dt
		
		conv_x_T = parameters['dead_time']+6*parameters['dead_time_sigma']
		dense_conv_x_nT = int(conv_x_T/_dt)+1
		conv_x_nT = int(conv_x_T/self.dp.dt)+1
		dense_conv_x = np.arange(0,dense_conv_x_nT)*_dt
		if parameters['dead_time_sigma']>0:
			dense_conv_val = normpdf(dense_conv_x,parameters['dead_time'],parameters['dead_time_sigma'])
			dense_conv_val[dense_conv_x<parameters['dead_time']] = 0.
		else:
			dense_conv_val = np.zeros_like(dense_conv_x)
			dense_conv_val[np.floor(parameters['dead_time']/_dt)] = 1.
		conv_x = np.arange(0,conv_x_nT)*self.dp.dt
		if must_downsample:
			#~ conv_val = average_downsample(dense_conv_val,conv_x_nT)
			if dense_conv_x_nT%conv_x_nT==0:
				ratio = int(np.round(dense_conv_x_nT/conv_x_nT))
			else:
				ratio = int(np.ceil(dense_conv_x_nT/conv_x_nT))
			tail = dense_conv_x_nT%ratio
			if tail!=0:
				padded_cv = np.concatenate((dense_conv_val,np.nan*np.ones(ratio-tail,dtype=np.float)),axis=0)
			else:
				padded_cv = dense_conv_val
			padded_cv = np.reshape(padded_cv,(-1,ratio))
			conv_val = np.nanmean(padded_cv,axis=1)
		else:
			conv_val = dense_conv_val
		conv_val/=np.sum(conv_val)
		return conv_val,conv_x
	
	def get_key(self,merge=None):
		experiment = self.experiment
		subject = self.subjectSession.get_name()
		session = self.subjectSession.get_session()
		if merge is None:
			key = "{experiment}_subject_{subject}_session_{session}".format(experiment=experiment,
					subject=subject,session=session)
		elif merge=='subjects':
			key = "{experiment}_session_{session}".format(experiment=experiment,session=session)
		elif merge=='sessions':
			key = "{experiment}_subject_{subject}".format(experiment=experiment,subject=subject)
		elif merge=='all':
			key = "{experiment}".format(experiment=experiment)
		return key
	
	def get_fitter_plot_handler(self,edges=None,merge=None,fit_output=None):
		if edges is None:
			edges = np.linspace(0,self.rt_cutoff,51)
		rt_edges = edges
		rt_centers = np.array([0.5*(e1+e0) for e1,e0 in zip(rt_edges[1:],rt_edges[:-1])])
		c_edges = np.linspace(0,1,self.confidence_partition+1)
		c_centers = np.array([0.5*(e1+e0) for e1,e0 in zip(c_edges[1:],c_edges[:-1])])
		dt = rt_edges[1]-rt_edges[0]
		dc = c_edges[1]-c_edges[0]
		hit = self.performance==1
		miss = np.logical_not(hit)
		subject_hit_histogram2d = np.histogram2d(self.rt[hit], self.confidence[hit], bins=[rt_edges,c_edges])[0].astype(np.float).T/dt
		subject_miss_histogram2d = np.histogram2d(self.rt[miss], self.confidence[miss], bins=[rt_edges,c_edges])[0].astype(np.float).T/dt
		
		subject_rt = np.array([np.sum(subject_hit_histogram2d,axis=0),
							   np.sum(subject_miss_histogram2d,axis=0)])
		subject_confidence = np.array([np.sum(subject_hit_histogram2d,axis=1),
									   np.sum(subject_miss_histogram2d,axis=1)])*dt
		
		self.set_fixed_parameters()
		model,t = self.theoretical_rt_confidence_distribution(fit_output)
		c = np.linspace(0,1,self.confidence_partition)
		model*=len(self.performance)
		model_hit_histogram2d = model[0]
		model_miss_histogram2d = model[1]
		model_rt = np.sum(model,axis=1)
		model_confidence = np.sum(model,axis=2)*self.dp.dt
		
		key = self.get_key(merge)
		output = {key:{'experimental':{'hit_histogram':{'x':rt_centers,'y':c_centers,'z':subject_hit_histogram2d},
									   'miss_histogram':{'x':rt_centers,'y':c_centers,'z':subject_miss_histogram2d},
									   'rt':{'x':rt_centers,'y':subject_rt},
									   'confidence':{'x':c_centers,'y':subject_confidence}},
						'theoretical':{'hit_histogram':{'x':t,'y':c,'z':model_hit_histogram2d},
									   'miss_histogram':{'x':t,'y':c,'z':model_miss_histogram2d},
									   'rt':{'x':t,'y':model_rt},
									   'confidence':{'x':c,'y':model_confidence}}}}
		return Fitter_plot_handler(output,self.time_units)
	
	def get_save_file_name(self):
		return Fitter_filename(experiment=self.experiment,method=self.method,name=self.subjectSession.get_name(),
				session=self.subjectSession.get_session(),optimizer=self.optimizer,suffix=self.suffix,
				confidence_map_method=self.high_confidence_mapping_method)
	
	# Defaults
	def default_decisionPolicyKwArgs(self):
		defaults = {'n':101,'prior_mu_var':self._prior_mu_var,'reward':1,'penalty':0}
		if hasattr(self,'_dt'):
			defaults['dt'] = self._dt
		else:
			if self.experiment=='Luminancia':
				if hasattr(self,'_ISI'):
					defaults['dt'] = self._ISI
				else:
					defaults['dt'] = 0.04
			else:
				if self.time_units=='seconds':
					defaults['dt'] = 0.001
				else:
					defaults['dt'] = 1.
		if hasattr(self,'_T'):
			defaults['T'] = self._T
		elif hasattr(self,'rt_cutoff'):
			defaults['T'] = self.rt_cutoff
		else:
			if self.time_units=='seconds':
				defaults['T'] = 10.
			else:
				defaults['T'] = 10000.
		if hasattr(self,'_iti'):
			defaults['iti'] = self._iti
		else:
			if self.time_units=='seconds':
				defaults['iti'] = 1.5
			else:
				defaults['iti'] = 1500.
		if hasattr(self,'_tp'):
			defaults['tp'] = self._tp
		else:
			defaults['tp'] = 0.
		if hasattr(self,'_model_var'):
			defaults['model_var'] = self._model_var
		else:
			if hasattr(self,'_ISI'):
				defaults['model_var'] = 50./self._ISI
			else:
				defaults['model_var'] = 1250.
		if hasattr(self,'_internal_var'):
			defaults['internal_var'] = self._internal_var
		else:
			defaults['internal_var'] = 0.
		if hasattr(self,'_fixed_stim_duration'):
			defaults['fixed_stim_duration'] = self._fixed_stim_duration
		else:
			defaults['fixed_stim_duration'] = 0.
		return defaults
	
	def default_fixed_parameters(self):
		return {}
	
	def default_start_point(self,forceCompute=False):
		try:
			if forceCompute:
				logging.debug('Forcing default start point recompute')
				raise Exception('Forcing recompute')
			return self.__default_start_point__
		except:
			if hasattr(self,'_fixed_parameters'):
				self.__default_start_point__ = self._fixed_parameters.copy()
			else:
				self.__default_start_point__ = {}
			if not 'cost' in self.__default_start_point__.keys() or self.__default_start_point__['cost'] is None:
				self.__default_start_point__['cost'] = 0.02 if self.time_units=='seconds' else 0.00002
			if not 'internal_var' in self.__default_start_point__.keys() or self.__default_start_point__['internal_var'] is None:
				try:
					from scipy.optimize import minimize
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						fun = lambda a: (np.mean(self.performance)-np.sum(self.mu_prob/(1.+np.exp(-0.596*self.mu/a))))**2
						res = minimize(fun,1000.,method='Nelder-Mead')
					self.__default_start_point__['internal_var'] = res.x[0]**2
				except Exception, e:
					logging.warning('Could not fit internal_var from data')
					self.__default_start_point__['internal_var'] = 1500. if self.time_units=='seconds' else 1.5
			if not 'phase_out_prob' in self.__default_start_point__.keys() or self.__default_start_point__['phase_out_prob'] is None:
				self.__default_start_point__['phase_out_prob'] = 0.05
			if not 'dead_time' in self.__default_start_point__.keys() or self.__default_start_point__['dead_time'] is None:
				dead_time = sorted(self.rt)[int(0.025*len(self.rt))]
				self.__default_start_point__['dead_time'] = dead_time
			if not 'confidence_map_slope' in self.__default_start_point__.keys() or self.__default_start_point__['confidence_map_slope'] is None:
				if self.high_confidence_mapping_method=='log_odds':
					self.__default_start_point__['confidence_map_slope'] = 17.2
				elif self.high_confidence_mapping_method=='belief':
					self.__default_start_point__['confidence_map_slope'] = 1.
				else:
					raise ValueError('Unknown high_confidence_mapping_method: {0}'.format(self.high_confidence_mapping_method))
			
			must_make_expensive_guess = ((not 'dead_time_sigma' in self.__default_start_point__.keys()) or\
										((not 'high_confidence_threshold' in self.__default_start_point__.keys()) and self.method!='full'))
			if must_make_expensive_guess:
				self.dp.set_cost(self.__default_start_point__['cost'])
				self.dp.set_internal_var(self.__default_start_point__['internal_var'])
				xub,xlb = self.dp.xbounds()
				first_passage_pdf = None
				for drift,drift_prob in zip(self.mu,self.mu_prob):
					gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
					if first_passage_pdf is None:
						first_passage_pdf = gs*drift_prob
					else:
						first_passage_pdf+= gs*drift_prob
				first_passage_pdf/=(np.sum(first_passage_pdf)*self.dp.dt)
				
				if not 'dead_time_sigma' in self.__default_start_point__.keys() or self.__default_start_point__['dead_time_sigma'] is None:
					mean_pdf = np.sum(first_passage_pdf*self.dp.t)*self.dp.dt
					var_pdf = np.sum(first_passage_pdf*(self.dp.t-mean_pdf)**2)*self.dp.dt
					var_rt = np.var(self.rt)
					min_dead_time_sigma_sp = 0.01 if self.time_units=='seconds' else 10.
					if var_rt-var_pdf>0:
						self.__default_start_point__['dead_time_sigma'] = np.max([np.sqrt(var_rt-var_pdf),min_dead_time_sigma_sp])
					else:
						self.__default_start_point__['dead_time_sigma'] = min_dead_time_sigma_sp
				
				rt_mode_ind = np.argmax(first_passage_pdf[0])
				rt_mode_ind+= 4 if self.dp.nT-rt_mode_ind>4 else 0
				log_odds = self.dp.log_odds()
				if not 'high_confidence_threshold' in self.__default_start_point__.keys() or self.__default_start_point__['high_confidence_threshold'] is None:
					self.__default_start_point__['high_confidence_threshold'] = log_odds[0][rt_mode_ind]
			else:
				if not 'high_confidence_threshold' in self.__default_start_point__.keys() or self.__default_start_point__['high_confidence_threshold'] is None:
					self.__default_start_point__['high_confidence_threshold'] = 0.3
			
			return self.__default_start_point__
	
	def default_bounds(self):
		if self.time_units=='seconds':
			defaults = {'cost':[0.,0.4],'dead_time':[0.,1.5],'dead_time_sigma':[0.001,6.]}
		else:
			defaults = {'cost':[0.,0.0004],'dead_time':[0.,1500.],'dead_time_sigma':[1.,6000.]}
		defaults['phase_out_prob'] = [0.,0.2]
		if self.high_confidence_mapping_method=='log_odds':
			defaults['high_confidence_threshold'] = [0.,np.log(self.dp.g[-1]/(1.-self.dp.g[-1]))]
		elif self.high_confidence_mapping_method=='belief':
			defaults['high_confidence_threshold'] = [0.,1.]
		else:
			raise ValueError('Unknown high_confidence_mapping_method: {0}'.format(self.high_confidence_mapping_method))
		defaults['confidence_map_slope'] = [0.,100.]
		default_sp = self.default_start_point()
		defaults['internal_var'] = [default_sp['internal_var']*0.2,default_sp['internal_var']*1.8]
		if default_sp['high_confidence_threshold']>defaults['high_confidence_threshold'][1]:
			defaults['high_confidence_threshold'][1] = 2*default_sp['high_confidence_threshold']
		return defaults
	
	def default_optimizer_kwargs(self):
		if self.optimizer=='cma':
			return {'restarts':1,'restart_from_best':'False'}
		elif self.optimizer=='basinhopping':
			return {'stepsize':0.25, 'minimizer_kwargs':{'method':'L-BFGS-B'}, take_step=None, accept_test=None, callback=None, interval=50, disp=False, niter_success=None}
		else:
			return {'disp': False, 'maxiter': 1000, 'maxfev': 10000, 'repetitions': 10}
	
	# Main fit method
	def fit(self,fixed_parameters={},start_point={},bounds={},optimizer_kwargs={},fit_arguments=None):
		if fit_arguments is None:
			self.set_fixed_parameters(fixed_parameters)
			self.set_start_point(start_point)
			self.set_bounds(bounds)
			self.fixed_parameters,self.fitted_parameters,start_point,bounds = self.sanitize_parameters_x0_bounds()
			
			self.set_optimizer_kwargs(optimizer_kwargs)
			self._fit_arguments = {'fixed_parameters':self.fixed_parameters,'fitted_parameters':self.fitted_parameters,\
								   'start_point':start_point,'bounds':bounds,'optimizer_kwargs':self.optimizer_kwargs}
		else:
			start_point = fit_arguments['start_point']
			bounds = fit_arguments['bounds']
			self.optimizer_kwargs = fit_arguments['optimizer_kwargs']
			self.fitted_parameters = fit_arguments['fitted_parameters']
			self.fixed_parameters = fit_arguments['fixed_parameters']
			self._fit_arguments = fit_arguments
		
		minimizer = self.init_minimizer(start_point,bounds,self.optimizer_kwargs)
		if self.experiment=='Luminancia':
			if self.method=='full':
				merit_function = self.full_merit
			elif self.method=='confidence_only':
				merit_function = self.confidence_only_merit
			elif self.method=='full_confidence':
				merit_function = self.full_confidence_merit
			elif self.method=='binary_confidence_only':
				merit_function = self.binary_confidence_only_merit
			elif self.method=='full_binary_confidence':
				merit_function = self.full_binary_confidence_merit
			else:
				raise ValueError('Unknown method "{0}" for experiment "{1}"'.format(self.method,self.experiment))
		elif self.experiment=='2AFC':
			if self.method=='full':
				merit_function = self.full_merit
			elif self.method=='confidence_only':
				merit_function = self.confidence_only_merit
			elif self.method=='full_confidence':
				merit_function = self.full_confidence_merit
			elif self.method=='binary_confidence_only':
				merit_function = self.binary_confidence_only_merit
			elif self.method=='full_binary_confidence':
				merit_function = self.full_binary_confidence_merit
			else:
				raise ValueError('Unknown method "{0}" for experiment "{1}"'.format(self.method,self.experiment))
		elif self.experiment=='Auditivo':
			if self.method=='full':
				merit_function = self.full_merit
			elif self.method=='confidence_only':
				merit_function = self.confidence_only_merit
			elif self.method=='full_confidence':
				merit_function = self.full_confidence_merit
			elif self.method=='binary_confidence_only':
				merit_function = self.binary_confidence_only_merit
			elif self.method=='full_binary_confidence':
				merit_function = self.full_binary_confidence_merit
			else:
				raise ValueError('Unknown method "{0}" for experiment "{1}"'.format(self.method,self.experiment))
		else:
			raise ValueError('Unknown experiment "{0}"'.format(self.experiment))
		self.__fit_internals__ = None
		self._fit_output = minimizer(merit_function)
		self.__fit_internals__ = None
		return self._fit_output
	
	# Savers
	def save(self):
		logging.debug('Fitter state that will be saved = "%s"',self.__getstate__())
		if not hasattr(self,'_fit_output'):
			raise ValueError('The Fitter instance has not performed any fit and still has no _fit_output attribute set')
		logging.info('Saving Fitter state to file "%s"',self.get_save_file_name())
		f = open(self.get_save_file_name(),'w')
		pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
		f.close()
	
	# Sanitizers
	def sanitize_parameters_x0_bounds(self):
		"""
		fitter.sanitize_parameters_x0_bounds()
		
		Some of the methods used to compute the merit assume certain
		parameters are fixed while others assume they are not. Furthermore
		some merit functions do not use all the parameters.
		This function allows the users to keep the flexibility of defining
		a single set of fixed parameters without worrying about the
		method specificities. The function takes the fixed_parameters,
		start_point and bounds specified by the user, and arranges them
		correctly for the specified merit method.
		
		Output:
		fixed_parameters,fitted_parameters,sanitized_start_point,sanitized_bounds
		
		fixed_parameters: A dict of parameter names as keys and their fixed values
		fitted_parameters: A list of the fitted parameters
		sanitized_start_point: A numpy ndarray with the fitted_parameters
		                       starting point
		sanitized_bounds: The fitted parameter's bounds. The specific
		                  format depends on the optimizer.
		
		"""
		_fixed_parameters = self._fixed_parameters.copy()
		for par in _fixed_parameters.keys():
			if _fixed_parameters[par] is None:
				_fixed_parameters[par] = self._start_point[par]
		
		binary_fixed_parameters = _fixed_parameters.copy()
		binary_fixed_parameters['confidence_map_slope'] = np.inf
		
		fittable_parameters = self.get_fittable_parameters()
		confidence_parameters = self.get_confidence_parameters()
		
		# Method specific fitted_parameters, fixed_parameters, starting points and bounds
		method_fitted_parameters = {'full_confidence':[],'full':[],'confidence_only':[],'binary_confidence_only':[],'full_binary_confidence':[]}
		method_fixed_parameters = {'full_confidence':_fixed_parameters.copy(),'full':_fixed_parameters.copy(),'confidence_only':_fixed_parameters.copy(),'binary_confidence_only':binary_fixed_parameters.copy(),'full_binary_confidence':binary_fixed_parameters.copy()}
		method_sp = {'full_confidence':[],'full':[],'confidence_only':[],'binary_confidence_only':[],'full_binary_confidence':[]}
		method_b = {'full_confidence':[],'full':[],'confidence_only':[],'binary_confidence_only':[],'full_binary_confidence':[]}
		for par in self._fitted_parameters:
			if par not in confidence_parameters:
				method_fitted_parameters['full'].append(par)
				method_sp['full'].append(self._start_point[par])
				method_b['full'].append(self._bounds[par])
				if par not in method_fixed_parameters['confidence_only'].keys():
					method_fixed_parameters['confidence_only'][par] = self._start_point[par]
				if par not in method_fixed_parameters['binary_confidence_only'].keys():
					method_fixed_parameters['binary_confidence_only'][par] = self._start_point[par]
				method_fitted_parameters['full_binary_confidence'].append(par)
				method_sp['full_binary_confidence'].append(self._start_point[par])
				method_b['full_binary_confidence'].append(self._bounds[par])
			else:
				method_fitted_parameters['confidence_only'].append(par)
				method_sp['confidence_only'].append(self._start_point[par])
				method_b['confidence_only'].append(self._bounds[par])
				if par not in method_fixed_parameters['full'].keys():
					method_fixed_parameters['full'][par] = self._start_point[par]
				if par!='confidence_map_slope':
					method_fitted_parameters['binary_confidence_only'].append(par)
					method_sp['binary_confidence_only'].append(self._start_point[par])
					method_b['binary_confidence_only'].append(self._bounds[par])
					method_fitted_parameters['full_binary_confidence'].append(par)
					method_sp['full_binary_confidence'].append(self._start_point[par])
					method_b['full_binary_confidence'].append(self._bounds[par])
			method_fitted_parameters['full_confidence'].append(par)
			method_sp['full_confidence'].append(self._start_point[par])
			method_b['full_confidence'].append(self._bounds[par])
		
		sanitized_start_point = np.array(method_sp[self.method])
		sanitized_bounds = list(np.array(method_b[self.method]).T)
		fitted_parameters = method_fitted_parameters[self.method]
		fixed_parameters = method_fixed_parameters[self.method]
		if len(fitted_parameters)==1 and self.optimizer=='cma':
			warnings.warn('CMA is unsuited for optimization of single dimensional parameter spaces. Optimizer was changed to Nelder-Mead')
			self.optimizer = 'Nelder-Mead'
		
		if self.optimizer!='cma':
			sanitized_bounds = [(lb,ub) for lb,ub in zip(sanitized_bounds[0],sanitized_bounds[1])]
		logging.debug('Sanitized fixed parameters = %s',fixed_parameters)
		logging.debug('Sanitized fitted parameters = %s',fitted_parameters)
		logging.debug('Sanitized start_point = %s',sanitized_start_point)
		logging.debug('Sanitized bounds = %s',sanitized_bounds)
		return (fixed_parameters,fitted_parameters,sanitized_start_point,sanitized_bounds)
	
	def sanitize_fmin_output(self,output,package='cma'):
		"""
		self.sanitize_fmin_output(output,package='cma')
		
		The cma package returns the fit output in one format while the
		scipy package returns it in a completely different way.
		This method returns the fit output in a common format:
		It returns a tuple out
		
		out[0]: A dictionary with the fitted parameter names as keys and
		        the values being the best fitting parameter value.
		out[1]: Merit function value
		out[2]: Number of function evaluations
		out[3]: Overall number of function evaluations (in the cma
		        package, these can be more if there is noise handling)
		out[4]: Number of iterations
		out[5]: Mean of the sample of solutions
		out[6]: Std of the sample of solutions
		
		"""
		logging.debug('Sanitizing minizer output with package: {0}'.format(package))
		logging.debug('Output to sanitize: {0}'.format(output))
		if package=='cma':
			fitted_x = {}
			for index,par in enumerate(self.fitted_parameters):
				fitted_x[par] = output[0][index]
			return (fitted_x,output[1:7])
		elif package=='scipy':
			fitted_x = {}
			for index,par in enumerate(self.fitted_parameters):
				fitted_x[par] = output['xbest'][index]
			return (fitted_x,output['funbest'],output['nfev'],output['nfev'],output['nit'],output['xmean'],output['xstd'])
		else:
			raise ValueError('Unknown package used for optimization. Unable to sanitize the fmin output')
	
	# Minimizer related methods
	def init_minimizer(self,start_point,bounds,optimizer_kwargs):
		logging.debug('init_minimizer args: start_point=%(start_point)s, bounds=%(bounds)s, optimizer_kwargs=%(optimizer_kwargs)s',{'start_point':start_point,'bounds':bounds,'optimizer_kwargs':optimizer_kwargs})
		if self.optimizer=='cma':
			scaling_factor = bounds[1]-bounds[0]
			logging.debug('scaling_factor = %s',scaling_factor)
			options = {'bounds':bounds,'CMA_stds':scaling_factor}
			options.update(optimizer_kwargs)
			restarts = options['restarts']
			del options['restarts']
			restart_from_best = options['restart_from_best']
			del options['restart_from_best']
			options = cma.CMAOptions(options)
			minimizer = lambda x: self.sanitize_fmin_output(cma.fmin(x,start_point,1./3.,options,restarts=restarts,restart_from_best=restart_from_best),package='cma')
			#~ minimizer = lambda x: self.sanitize_fmin_output((start_point,None,None,None,None,None,None,None),'cma')
		elif self.optimizer=='basinhopping':
			options = optimizer_kwargs.copy()
			for k in options:
				if k not in ['niter','T','stepsize','minimizer_kwargs','take_step','accept_test','callback','interval','disp','niter_success']:
					del options[k]
				#~ elif k=='stepsize':
					#~ options[k] = options[k]*(bounds[1]-bounds[0])
			lambda x: self.sanitize_fmin_output(scipy.basinhopping(x, start_point, **options),package='scipy')
		else:
			repetitions = optimizer_kwargs['repetitions']
			_start_points = [start_point]
			for rsp in np.random.rand(repetitions-1,len(start_point)):
				temp = []
				for val,(lb,ub) in zip(rsp,bounds):
					temp.append(val*(ub-lb)+lb)
				_start_points.append(np.array(temp))
			logging.debug('Array of start_points = {0}',_start_points)
			start_point_generator = iter(_start_points)
			minimizer = lambda x: self.sanitize_fmin_output(self.repeat_minimize(x,start_point_generator,bounds=bounds,optimizer_kwargs=optimizer_kwargs),package='scipy')
			#~ minimizer = lambda x: self.sanitize_fmin_output({'xbest':start_point,'funbest':None,'nfev':None,'nit':None,'xmean':None,'xstd':None},'scipy')
		
		return minimizer
	
	def repeat_minimize(self,merit,start_point_generator,bounds,optimizer_kwargs):
		output = {'xs':[],'funs':[],'nfev':0,'nit':0,'xbest':None,'funbest':None,'xmean':None,'xstd':None,'funmean':None,'funstd':None}
		repetitions = 0
		for start_point in start_point_generator:
			repetitions+=1
			logging.debug('Round {2} with start_point={0} and bounds={1}'.format(start_point, bounds,repetitions))
			res = scipy.optimize.minimize(merit,start_point, method=self.optimizer,bounds=bounds,options=optimizer_kwargs)
			logging.debug('New round with start_point={0} and bounds={0}'.format(start_point, bounds))
			logging.debug('Round {0} ended. Fun val: {1}. x={2}'.format(repetitions,res.fun,res.x))
			output['xs'].append(res.x)
			output['funs'].append(res.fun)
			output['nfev']+=res.nfev
			output['nit']+=res.nit
			if output['funbest'] is None or res.fun<output['funbest']:
				output['funbest'] = res.fun
				output['xbest'] = res.x
			logging.debug('Best so far: {0} at point {1}'.format(output['funbest'],output['xbest']))
		arr_xs = np.array(output['xs'])
		arr_funs = np.array(output['funs'])
		output['xmean'] = np.mean(arr_xs)
		output['xstd'] = np.std(arr_xs)
		output['funmean'] = np.mean(arr_funs)
		output['funstd'] = np.std(arr_funs)
		return output
	
	# Auxiliary method
	def high_confidence_mapping(self,high_confidence_threshold,confidence_map_slope):
		if self.high_confidence_mapping_method=='log_odds':
			return self.high_confidence_mapping_log_odds(high_confidence_threshold,confidence_map_slope)
		elif self.high_confidence_mapping_method=='belief':
			return self.high_confidence_mapping_belief(high_confidence_threshold,confidence_map_slope)
		else:
			raise ValueError('Undefined high confidence mapping method: {0}'.format(self.high_confidence_mapping_method))
	
	def high_confidence_mapping_log_odds(self,high_confidence_threshold,confidence_map_slope):
		if self.time_units=='seconds' and self.dp.dt>1e-3:
			_dt = 1e-3
		elif self.time_units=='milliseconds' and self.dp.dt>1.:
			_dt = 1.
		else:
			_dt = None
		
		if _dt:
			_nT = int(self.dp.T/_dt)+1
			_t = np.arange(0.,_nT,dtype=np.float64)*_dt
			log_odds = self.dp.log_odds()
			log_odds = np.array([np.interp(_t,self.dp.t,log_odds[0]),np.interp(_t,self.dp.t,log_odds[1])])
		else:
			_nT = self.dp.nT
			log_odds = self.dp.log_odds()
			_dt = self.dp.dt
		# Likely to raise warnings with exp overflows or invalid values in multiply
		# if confidence_map_slope is inf or log_odds==high_confidence_threshold
		# These issues are resolved naturally in the two-line statements
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			phigh = 1./(1.+np.exp(confidence_map_slope*(high_confidence_threshold-log_odds)))
		phigh[high_confidence_threshold==log_odds] = 0.5
		
		if _dt:
			if _nT%self.dp.nT==0:
				ratio = int(np.round(_nT/self.dp.nT))
			else:
				ratio = int(np.ceil(_nT/self.dp.nT))
			tail = _nT%ratio
			if tail!=0:
				padded_phigh = np.concatenate((phigh,np.nan*np.ones((2,ratio-tail),dtype=np.float)),axis=1)
			else:
				padded_phigh = phigh
			padded_phigh = np.reshape(padded_phigh,(2,-1,ratio))
			phigh = np.nanmean(padded_phigh,axis=2)
		return phigh
	
	def high_confidence_mapping_belief(self,high_confidence_threshold,confidence_map_slope):
		if self.time_units=='seconds' and self.dp.dt>1e-3:
			_dt = 1e-3
		elif self.time_units=='milliseconds' and self.dp.dt>1.:
			_dt = 1.
		else:
			_dt = None
		
		if _dt:
			_nT = int(self.dp.T/_dt)+1
			_t = np.arange(0.,_nT,dtype=np.float64)*_dt
			belief = self.dp.bounds.copy()
			belief = np.array([np.interp(_t,self.dp.t,2*(belief[0]-0.5)),np.interp(_t,self.dp.t,2*(0.5-belief[1]))])
		else:
			_nT = self.dp.nT
			belief = self.dp.bounds.copy()
			belief[0] = 2*(belief[0]-0.5)
			belief[1] = 2*(0.5-belief[1])
			_dt = self.dp.dt
		
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			phigh = confidence_map_slope*(belief-high_confidence_threshold)
		phigh[phigh>1] = 1
		plow[plow<0] = 0
		phigh[high_confidence_threshold==belief] = 0.5
		
		if _dt:
			if _nT%self.dp.nT==0:
				ratio = int(np.round(_nT/self.dp.nT))
			else:
				ratio = int(np.ceil(_nT/self.dp.nT))
			tail = _nT%ratio
			if tail!=0:
				padded_phigh = np.concatenate((phigh,np.nan*np.ones((2,ratio-tail),dtype=np.float)),axis=1)
			else:
				padded_phigh = phigh
			padded_phigh = np.reshape(padded_phigh,(2,-1,ratio))
			phigh = np.nanmean(padded_phigh,axis=2)
		return phigh
	
	def confidence_mapping_pdf_matrix(self,first_passage_pdfs,parameters,mapped_confidences=None,return_unconvoluted_matrix=False,dead_time_convolver=None):
		indeces = np.arange(0,self.confidence_partition,dtype=np.float)
		confidence_array = np.linspace(0,1,self.confidence_partition)
		nT = self.dp.nT
		confidence_matrix = np.zeros((2,self.confidence_partition,nT))
		if mapped_confidences is None:
			mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		try:
			conv_val,conv_x = dead_time_convolver
		except:
			conv_val,conv_x = self.get_dead_time_convolver(parameters)
		_nT = len(conv_x)
		for decision_ind,(first_passage_pdf,mapped_confidence) in enumerate(zip(first_passage_pdfs,mapped_confidences)):
			cv_inds = np.interp(mapped_confidence,confidence_array,indeces)
			for index,(cv_ind,floor_cv_ind,ceil_cv_ind,fppdf) in enumerate(zip(cv_inds,np.floor(cv_inds).astype(np.int),np.ceil(cv_inds).astype(np.int),first_passage_pdf)):
				norm = 0.
				if index==0:
					weight = 1.-np.mod(cv_ind,1)
					confidence_matrix[decision_ind,floor_cv_ind,index] = fppdf*weight
					confidence_matrix[decision_ind,ceil_cv_ind,index]+= fppdf*(1.-weight)
					prev_norm = fppdf
				else:
					if np.abs(cv_ind-prior_cv_ind)<=1.5:
						weight = 1.-np.mod(cv_ind,1)
						confidence_matrix[decision_ind,floor_cv_ind,index] = fppdf*weight
						confidence_matrix[decision_ind,ceil_cv_ind,index]+= fppdf*(1.-weight)
						norm = fppdf
					else:
						if prior_cv_ind<cv_ind:
							prev_polypars = np.polyfit([prior_ceil_cv_ind,ceil_cv_ind],[prior_fppdf,0.],1)
							curr_polypars = np.polyfit([prior_ceil_cv_ind,ceil_cv_ind],[0.,fppdf],1)
							
							prev_temp_fppdf = np.polyval(prev_polypars,np.arange(prior_ceil_cv_ind+1,ceil_cv_ind+1,dtype=np.float))
							curr_temp_fppdf = np.polyval(curr_polypars,np.arange(prior_ceil_cv_ind+1,ceil_cv_ind+1,dtype=np.float))
							prev_temp_fppdf[prev_temp_fppdf<0.] = 0.
							curr_temp_fppdf[curr_temp_fppdf<0.] = 0.
							
							prev_norm+= np.sum(prev_temp_fppdf)
							norm = np.sum(curr_temp_fppdf)
							confidence_matrix[decision_ind,prior_ceil_cv_ind+1:ceil_cv_ind+1,index-1]+= prev_temp_fppdf
							confidence_matrix[decision_ind,prior_ceil_cv_ind+1:ceil_cv_ind+1,index] = curr_temp_fppdf
						else:
							prev_polypars = np.polyfit([prior_cv_ind,floor_cv_ind],[prior_fppdf,0.],1)
							curr_polypars = np.polyfit([prior_cv_ind,floor_cv_ind],[0.,fppdf],1)
							
							prev_temp_fppdf = np.polyval(prev_polypars,np.arange(prior_floor_cv_ind-1,floor_cv_ind-1,-1,dtype=np.float))
							curr_temp_fppdf = np.polyval(curr_polypars,np.arange(prior_floor_cv_ind-1,floor_cv_ind-1,-1,dtype=np.float))
							prev_temp_fppdf[prev_temp_fppdf<0.] = 0.
							curr_temp_fppdf[curr_temp_fppdf<0.] = 0.
							
							prev_norm+= np.sum(prev_temp_fppdf)
							norm = np.sum(curr_temp_fppdf)
							if floor_cv_ind>0:
								confidence_matrix[decision_ind,prior_floor_cv_ind-1:floor_cv_ind-1:-1,index-1]+= prev_temp_fppdf
								confidence_matrix[decision_ind,prior_floor_cv_ind-1:floor_cv_ind-1:-1,index] = curr_temp_fppdf
							else:
								confidence_matrix[decision_ind,prior_floor_cv_ind-1::-1,index-1]+= prev_temp_fppdf
								confidence_matrix[decision_ind,prior_floor_cv_ind-1::-1,index] = curr_temp_fppdf
					if prev_norm>0:
						confidence_matrix[decision_ind,:,index-1]*=prior_fppdf/prev_norm
					if index==len(cv_inds)-1 and norm>0:
						confidence_matrix[decision_ind,:,index]*=fppdf/norm
					prev_norm = norm
				prior_fppdf = fppdf
				prior_cv_ind = cv_ind
				prior_floor_cv_ind = floor_cv_ind
				prior_ceil_cv_ind = ceil_cv_ind
		conv_confidence_matrix = scipy.signal.fftconvolve(confidence_matrix,conv_val.reshape((1,1,-1)),mode='full')
		conv_confidence_matrix[conv_confidence_matrix<0] = 0.
		conv_confidence_matrix/=(np.sum(conv_confidence_matrix)*self.dp.dt)
		if return_unconvoluted_matrix:
			confidence_matrix/=(np.sum(confidence_matrix)*self.dp.dt)
			output = (conv_confidence_matrix,confidence_matrix)
		else:
			output = conv_confidence_matrix
		return output
	
	def decision_pdf(self,first_passage_pdfs,parameters,dead_time_convolver=None):
		try:
			conv_val,conv_x = dead_time_convolver
		except:
			conv_val,conv_x = self.get_dead_time_convolver(parameters)
		decision_pdfs = scipy.signal.fftconvolve(first_passage_pdfs,conv_val.reshape((1,-1)),mode='full')
		decision_pdfs[confidence_pdfs<0] = 0.
		decision_pdfs/=(np.sum(decision_pdfs)*self.dp.dt)
		return decision_pdfs
	
	def binary_confidence_pdf(self,first_passage_pdfs,parameters,phigh_low=None,dead_time_convolver=None):
		if phigh_low is None:
			phigh = self.high_confidence_mapping(parameters['high_confidence_threshold'],np.inf)
			plow = 1.-phigh
		else:
			phigh,plow = phigh_low
		try:
			conv_val,conv_x = dead_time_convolver
		except:
			conv_val,conv_x = self.get_dead_time_convolver(parameters)
		confidence_rt = np.concatenate((np.array(first_passage_pdfs)[:,None,:]*plow[:,None,:],np.array(first_passage_pdfs)[:,None,:]*phigh[:,None,:]),axis=1)
		confidence_pdfs = scipy.signal.fftconvolve(confidence_rt,conv_val.reshape((1,1,-1)),mode='full')
		confidence_pdfs[confidence_pdfs<0] = 0.
		confidence_pdfs/=(np.sum(confidence_pdfs)*self.dp.dt)
		
		return confidence_pdfs
	
	# Method dependent merits
	def full_merit(self,x):
		parameters = self.get_parameters_dict_from_array(x)
		nlog_likelihood = 0.
		if 'cost' in self.get_fitted_parameters() or 'internal_var' in self.get_fitted_parameters():
			self.dp.set_cost(parameters['cost'])
			self.dp.set_internal_var(parameters['internal_var'])
			must_compute_first_passage_time = True
			must_store_first_passage_time = False
			xub,xlb = self.dp.xbounds()
		else:
			if self.__fit_internals__ is None:
				must_compute_first_passage_time = True
				must_store_first_passage_time = True
				self.dp.set_cost(parameters['cost'])
				self.dp.set_internal_var(parameters['internal_var'])
				xub,xlb = self.dp.xbounds()
				self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{}}
			else:
				must_compute_first_passage_time = False
				first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		if 'dead_time' in self.get_fitted_parameters() or 'dead_time_sigma' in self.get_fitted_parameters():
			dead_time_convolver = self.get_dead_time_convolver(parameters)
		else:
			if self.__fit_internals__ is None:
				dead_time_convolver = self.get_dead_time_convolver(parameters)
				self.__fit_internals__ = {'dead_time_convolver':dead_time_convolver}
			else:
				try:
					dead_time_convolver = self.__fit_internals__['dead_time_convolver']
				except:
					dead_time_convolver = self.get_dead_time_convolver(parameters)
					self.__fit_internals__['dead_time_convolver'] = dead_time_convolver
		for index,drift in enumerate(self.mu):
			if must_compute_first_passage_time:
				first_passage_time = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
				if must_store_first_passage_time:
					self.__fit_internals__['first_passage_times'][drift] = first_passage_time
			else:
				first_passage_time = self.__fit_internals__['first_passage_times'][drift]
			gs = self.decision_pdf(first_passage_time,parameters,dead_time_convolver=dead_time_convolver)
			t = np.arange(0,gs.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
				nlog_likelihood-= np.log(rt_likelihood(t,gs[1-int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def confidence_only_merit(self,x):
		parameters = self.get_parameters_dict_from_array(x)
		nlog_likelihood = 0.
		if self.__fit_internals__ is None:
			self.dp.set_cost(parameters['cost'])
			self.dp.set_internal_var(parameters['internal_var'])
			xub,xlb = self.dp.xbounds()
			must_compute_first_passage_time = True
			self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{}}
		else:
			must_compute_first_passage_time = False
			first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/float(self.confidence_partition)
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		
		for index,drift in enumerate(self.mu):
			if must_compute_first_passage_time:
				gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
				self.__fit_internals__['first_passage_times'][drift] = gs
			else:
				gs = self.__fit_internals__['first_passage_times'][drift]
			conf_lik_pdf = np.sum(self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences,dead_time_convolver=dead_time_convolver),axis=2)
			indeces = self.mu_indeces==index
			for perf,conf in zip(self.performance[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(confidence_likelihood(conf_lik_pdf[1-int(perf)],conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def full_confidence_merit(self,x):
		parameters = self.get_parameters_dict_from_array(x)
		nlog_likelihood = 0.
		if 'cost' in self.get_fitted_parameters() or 'internal_var' in self.get_fitted_parameters():
			self.dp.set_cost(parameters['cost'])
			self.dp.set_internal_var(parameters['internal_var'])
			must_compute_first_passage_time = True
			must_store_first_passage_time = False
			xub,xlb = self.dp.xbounds()
		else:
			if self.__fit_internals__ is None:
				must_compute_first_passage_time = True
				must_store_first_passage_time = True
				self.dp.set_cost(parameters['cost'])
				self.dp.set_internal_var(parameters['internal_var'])
				xub,xlb = self.dp.xbounds()
				self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{}}
			else:
				must_compute_first_passage_time = False
				first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)/float(self.confidence_partition)
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		if 'dead_time' in self.get_fitted_parameters() or 'dead_time_sigma' in self.get_fitted_parameters():
			dead_time_convolver = self.get_dead_time_convolver(parameters)
		else:
			if self.__fit_internals__ is None:
				dead_time_convolver = self.get_dead_time_convolver(parameters)
				self.__fit_internals__ = {'dead_time_convolver':dead_time_convolver}
			else:
				try:
					dead_time_convolver = self.__fit_internals__['dead_time_convolver']
				except:
					dead_time_convolver = self.get_dead_time_convolver(parameters)
					self.__fit_internals__['dead_time_convolver'] = dead_time_convolver
		for index,drift in enumerate(self.mu):
			if must_compute_first_passage_time:
				gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
				if must_store_first_passage_time:
					self.__fit_internals__['first_passage_times'][drift] = gs
			else:
				gs = self.__fit_internals__['first_passage_times'][drift]
			rt_conf_lik_matrix = self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences,dead_time_convolver=dead_time_convolver)
			t = np.arange(0,rt_conf_lik_matrix.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,rt_conf_lik_matrix[1-int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def full_binary_confidence_merit(self,x):
		parameters = self.get_parameters_dict_from_array(x)
		nlog_likelihood = 0.
		if 'cost' in self.get_fitted_parameters() or 'internal_var' in self.get_fitted_parameters():
			self.dp.set_cost(parameters['cost'])
			self.dp.set_internal_var(parameters['internal_var'])
			must_compute_first_passage_time = True
			must_store_first_passage_time = False
			xub,xlb = self.dp.xbounds()
		else:
			if self.__fit_internals__ is None:
				must_compute_first_passage_time = True
				must_store_first_passage_time = True
				self.dp.set_cost(parameters['cost'])
				self.dp.set_internal_var(parameters['internal_var'])
				xub,xlb = self.dp.xbounds()
				self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{}}
			else:
				must_compute_first_passage_time = False
				first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.25*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		phigh = self.high_confidence_mapping(parameters['high_confidence_threshold'],np.inf)
		phigh_low = (phigh,1.-phigh)
		if 'dead_time' in self.get_fitted_parameters() or 'dead_time_sigma' in self.get_fitted_parameters():
			dead_time_convolver = self.get_dead_time_convolver(parameters)
		else:
			if self.__fit_internals__ is None:
				dead_time_convolver = self.get_dead_time_convolver(parameters)
				self.__fit_internals__ = {'dead_time_convolver':dead_time_convolver}
			else:
				try:
					dead_time_convolver = self.__fit_internals__['dead_time_convolver']
				except:
					dead_time_convolver = self.get_dead_time_convolver(parameters)
					self.__fit_internals__['dead_time_convolver'] = dead_time_convolver
		median_confidence = np.median(self.confidence)
		for index,drift in enumerate(self.mu):
			if must_compute_first_passage_time:
				gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
				if must_store_first_passage_time:
					self.__fit_internals__['first_passage_times'][drift] = gs
			else:
				gs = self.__fit_internals__['first_passage_times'][drift]
			binary_confidence_pdf = self.binary_confidence_pdf(gs,parameters,phigh_low=phigh_low,dead_time_convolver=dead_time_convolver)
			t = np.arange(0,binary_confidence_pdf.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				binary_conf = 0 if conf<median_confidence else 1
				nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_pdf[(1-int(perf)),binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def binary_confidence_only_merit(self,x):
		parameters = self.get_parameters_dict_from_array(x)
		nlog_likelihood = 0.
		if self.__fit_internals__ is None:
			self.dp.set_cost(parameters['cost'])
			self.dp.set_internal_var(parameters['internal_var'])
			xub,xlb = self.dp.xbounds()
			must_compute_first_passage_time = True
			self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{},'dead_time_convolver': self.get_dead_time_convolver(parameters)}
		else:
			must_compute_first_passage_time = False
			first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.5*parameters['phase_out_prob']
		phigh = self.high_confidence_mapping(parameters['high_confidence_threshold'],np.inf)
		phigh_low = (phigh,1.-phigh)
		dead_time_convolver = self.__fit_internals__['dead_time_convolver']
		median_confidence = np.median(self.confidence)
		
		for index,drift in enumerate(self.mu):
			if must_compute_first_passage_time:
				gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
				self.__fit_internals__['first_passage_times'][drift] = gs
			else:
				gs = self.__fit_internals__['first_passage_times'][drift]
			binary_confidence_pdf = np.sum(self.binary_confidence_pdf(gs,parameters,phigh_low=phigh_low,dead_time_convolver=dead_time_convolver),axis=0)
			t = np.arange(0,binary_confidence_pdf.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,conf in zip(self.rt[indeces],self.confidence[indeces]):
				binary_conf = 0 if conf<median_confidence else 1
				nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_pdf[binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	# Force to compute merit functions on an arbitrary parameter dict
	def forced_compute_full_merit(self,parameters):
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		for index,drift in enumerate(self.mu):
			gs = self.decision_pdf(np.array(self.dp.rt(drift,bounds=(xub,xlb))),parameters,dead_time_convolver=dead_time_convolver)
			t = np.arange(0,gs.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
				nlog_likelihood-= np.log(rt_likelihood(t,gs[1-int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def forced_compute_confidence_only_merit(self,parameters):
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/float(self.confidence_partition)
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			conf_lik_pdf = np.sum(self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences,dead_time_convolver=dead_time_convolver),axis=2)
			indeces = self.mu_indeces==index
			for perf,conf in zip(self.performance[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(confidence_likelihood(conf_lik_pdf[1-int(perf)],conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def forced_compute_full_confidence_merit(self,parameters):
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)/float(self.confidence_partition)
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			rt_conf_lik_matrix = self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences,dead_time_convolver=dead_time_convolver)
			t = np.arange(0,rt_conf_lik_matrix.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,rt_conf_lik_matrix[1-int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def forced_compute_full_binary_confidence_merit(self,parameters):
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.25*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		phigh = self.high_confidence_mapping(parameters['high_confidence_threshold'],np.inf)
		phigh_low = (phigh,1.-phigh)
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		median_confidence = np.median(self.confidence)
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			binary_confidence_pdf = self.binary_confidence_pdf(gs,parameters,phigh_low=phigh_low,dead_time_convolver=dead_time_convolver)
			t = np.arange(0,rt_conf_lik_matrix.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				binary_conf = 0 if conf<median_confidence else 1
				nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_pdf[(1-int(perf)),binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def forced_compute_binary_confidence_only_merit(self,parameters):
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		dead_time_convolver = self.get_dead_time_convolver(parameters)
		random_rt_likelihood = 0.5*parameters['phase_out_prob']
		phigh = self.high_confidence_mapping(parameters['high_confidence_threshold'],np.inf)
		phigh_low = (phigh,1.-phigh)
		median_confidence = np.median(self.confidence)
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			binary_confidence_pdf = np.sum(self.binary_confidence_pdf(gs,parameters,phigh_low=phigh_low,dead_time_convolver=dead_time_convolver),axis=0)
			t = np.arange(0,rt_conf_lik_matrix.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,conf in zip(self.rt[indeces],self.confidence[indeces]):
				binary_conf = 0 if conf<median_confidence else 1
				nlog_likelihood-=np.log(rt_likelihood(t,binary_confidence_pdf[binary_conf],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	# Theoretical predictions
	def theoretical_rt_confidence_distribution(self,fit_output=None,binary_confidence=None):
		if binary_confidence is None:
			binary_confidence = self.method=='binary_confidence_only' or self.method=='full_binary_confidence'
		parameters = self.get_parameters_dict_from_fit_output(fit_output)
		
		self.dp.set_cost(parameters['cost'])
		if self.experiment!='Luminancia':
			self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		
		if binary_confidence:
			phigh = self.high_confidence_mapping(parameters['high_confidence_threshold'],np.inf)
			plow = 1.-phigh
		else:
			mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		output = None
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			if binary_confidence:
				rt_conf_lik_matrix = self.binary_confidence_pdf(gs,parameters,phigh_low=(phigh,plow))
			else:
				rt_conf_lik_matrix = self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences)
			
			if output is None:
				output = rt_conf_lik_matrix*self.mu_prob[index]
			else:
				output+= rt_conf_lik_matrix*self.mu_prob[index]
		output/=(np.sum(output)*self.dp.dt)
		t = np.arange(0,output.shape[2],dtype=np.float)*self.dp.dt
		#~ plt.subplot(211)
		#~ plt.plot(t,output[0,0],label='[0,0] hit low')
		#~ plt.plot(t,output[0,1],label='[0,1] hit high')
		#~ plt.plot(t,output[1,0],label='[1,0] miss low')
		#~ plt.plot(t,output[1,1],label='[1,1] miss high')
		#~ plt.legend()
		#~ plt.subplot(212)
		#~ plt.plot(t,np.sum(output,axis=0)[0],label='sum(..,axis=0)[0] low')
		#~ plt.plot(t,np.sum(output,axis=0)[1],label='sum(..,axis=0)[1] high')
		#~ plt.legend()
		#~ plt.show(True)
		random_rt_likelihood = np.ones_like(output)
		random_rt_likelihood[:,:,np.logical_or(t<self.min_RT,t>self.max_RT)] = 0.
		random_rt_likelihood/=(np.sum(random_rt_likelihood)*self.dp.dt)
		return output*(1.-parameters['phase_out_prob'])+parameters['phase_out_prob']*random_rt_likelihood, np.arange(0,output.shape[2],dtype=np.float)*self.dp.dt
	
	# Plotter
	def plot_fit(self,fit_output=None,saver=None,show=True):
		if not can_plot:
			raise ImportError('Could not import matplotlib package and it is imposible to plot fit')
		
		self.get_fitter_plot_handler(fit_output=fit_output).plot(saver=saver,show=show)

class Fitter_plot_handler():
	def __init__(self,obj,time_units):
		self._time_units = time_units
		self.required_data = ['hit_histogram','miss_histogram','rt','confidence','t_array','c_array']
		# For backward compatibility
		new_style_required = ['t_array','c_array']
		self.categories = ['experimental','theoretical']
		self.dictionary = {}
		try:
			for key in obj.keys():
				self.dictionary[key] = {}
				for category in self.categories:
					self.dictionary[key][category] = {}
					if any([not req in obj[key][category] for req in new_style_required]):
						# Old style handler
						self.dictionary[key][category]['t_array'] = copy.deepcopy(obj[key][category]['hit_histogram']['x'])
						self.dictionary[key][category]['c_array'] = copy.deepcopy(obj[key][category]['hit_histogram']['y'])
						self.dictionary[key][category]['hit_histogram'] = copy.deepcopy(obj[key][category]['hit_histogram']['z'])
						self.dictionary[key][category]['miss_histogram'] = copy.deepcopy(obj[key][category]['miss_histogram']['z'])
						self.dictionary[key][category]['rt'] = copy.deepcopy(obj[key][category]['rt']['y'])
						self.dictionary[key][category]['confidence'] = copy.deepcopy(obj[key][category]['confidence']['y'])
					else:
						# New style handler
						for required_data in self.required_data:
							self.dictionary[key][category][required_data] = copy.deepcopy(obj[key][category][required_data])
		except:
			raise RuntimeError('Invalid object used to init Fitter_plot_handler')
	
	def time_units(self):
		return {'seconds':'s','milliseconds':'ms'}[self._time_units]
	
	def keys(self):
		return self.dictionary.keys()
	
	def __getitem__(self,key):
		return self.dictionary[key]
	
	def __setitem__(self,key,value):
		self.dictionary[key] = value
	
	def __aliased_iadd__(self,other,key_aliaser=lambda key:key):
		for other_key in other.keys():
			if key_aliaser(other_key) in self.keys():
				for category in self.categories:
					torig = self[key_aliaser(other_key)][category]['t_array']
					tadded = other[other_key][category]['t_array']
					for required_data in ['hit_histogram','miss_histogram','rt']:
						orig = self[key_aliaser(other_key)][category][required_data]
						added = other[other_key][category][required_data]
						if len(torig)<len(tadded):
							summed = added+np.pad(orig, ((0,0),(0,len(tadded)-len(torig))),'constant',constant_values=(0., 0.))
							self[key_aliaser(other_key)][category][required_data] = summed
						elif len(torig)>len(tadded):
							summed = orig+np.pad(added, ((0,0),(0,len(torig)-len(tadded))),'constant',constant_values=(0., 0.))
							self[key_aliaser(other_key)][category][required_data] = summed
						else:
							self[key_aliaser(other_key)][category][required_data]+= added
					if len(torig)<len(tadded):
						self[key_aliaser(other_key)][category]['t_array'] = copy.copy(tadded)
					self[key_aliaser(other_key)][category]['confidence']+= \
							other[other_key][category]['confidence']
			else:
				self[key_aliaser(other_key)] = {}
				for category in self.categories:
					self[key_aliaser(other_key)][category] = {}
					for required_data in self.required_data:
						self[key_aliaser(other_key)][category][required_data] = \
							copy.deepcopy(other[other_key][category][required_data])
		return self
	
	def __iadd__(self,other):
		return self.__aliased_iadd__(other)
	
	def __add__(self,other):
		output = Fitter_plot_handler(self)
		output+=other
		return output
	
	def normalize(self):
		for key in self.keys():
			for category in self.categories:
				dt = self[key][category]['t_array'][1]-self[key][category]['t_array'][0]
				for required_data in self.required_data:
					if not required_data in ['t_array','c_array']:
						if required_data!='confidence':
							self[key][category][required_data]/= (np.sum(self[key][category][required_data])*dt)
						else:
							self[key][category][required_data]/= np.sum(self[key][category][required_data])
	
	def plot(self,saver=None,show=True,xlim_rt_cutoff=True,fig=None,logscale=True):
		if not can_plot:
			raise ImportError('Could not import matplotlib package and it is imposible to produce any plot')
		self.normalize()
		
		for key in sorted(self.keys()):
			logging.info('Preparing to plot key {0}'.format(key))
			subj = self.dictionary[key]['experimental']
			model = self.dictionary[key]['theoretical']
			
			rt_cutoff = subj['t_array'][-1]+0.5*(subj['t_array'][-1]-subj['t_array'][-2])
			
			if fig is None:
				fig = plt.figure(figsize=(10,12))
				logging.debug('Created figure instance {0}'.format(fig.number))
			else:
				logging.debug('Will use figure instance {0}'.format(fig.number))
				fig.clf()
				plt.figure(fig.number)
				logging.debug('Cleared figure instance {0} and setted it as the current figure'.format(fig.number))
			gs1 = gridspec.GridSpec(1, 2,left=0.10, right=0.90, top=0.95,bottom=0.70)
			gs2 = gridspec.GridSpec(2, 2,left=0.10, right=0.85, wspace=0.05, hspace=0.05, top=0.62,bottom=0.05)
			gs3 = gridspec.GridSpec(1, 1,left=0.87, right=0.90, wspace=0.1, top=0.62,bottom=0.05)
			logging.debug('Created gridspecs')
			axrt = plt.subplot(gs1[0])
			logging.debug('Created rt axes')
			plt.step(subj['t_array'],subj['rt'][0],'b',label='Subject hit')
			#~ plt.step(subj['t_array'],-subj['rt'][1],'r',label='Subject miss')
			plt.step(subj['t_array'],subj['rt'][1],'r',label='Subject miss')
			plt.plot(model['t_array'],model['rt'][0],'b',label='Model hit',linewidth=3)
			#~ plt.plot(model['t_array'],-model['rt'][1],'r',label='Model miss',linewidth=3)
			plt.plot(model['t_array'],model['rt'][1],'r',label='Model miss',linewidth=3)
			logging.debug('Plotted rt axes')
			if xlim_rt_cutoff:
				axrt.set_xlim([0,rt_cutoff])
			plt.xlabel('RT [{time_units}]'.format(time_units=self.time_units()))
			plt.ylabel('Prob density')
			plt.legend(loc='best', fancybox=True, framealpha=0.5)
			logging.debug('Completed rt axes plot, legend and labels')
			axconf = plt.subplot(gs1[1])
			logging.debug('Created confidence axes')
			plt.step(subj['c_array'],subj['confidence'][0],'b',label='Subject hit')
			plt.plot(model['c_array'],model['confidence'][0],'b',label='Model hit',linewidth=3)
			if logscale:
				plt.step(subj['c_array'],subj['confidence'][1],'r',label='Subject miss')
				plt.plot(model['c_array'],model['confidence'][1],'r',label='Model miss',linewidth=3)
				axconf.set_yscale('log')
			else:
				#~ plt.step(subj['c_array'],-subj['confidence'][1],'r',label='Subject miss')
				#~ plt.plot(model['c_array'],-model['confidence'][1],'r',label='Model miss',linewidth=3)
				plt.step(subj['c_array'],subj['confidence'][1],'r',label='Subject miss')
				plt.plot(model['c_array'],model['confidence'][1],'r',label='Model miss',linewidth=3)
			logging.debug('Plotted confidence axes')
			plt.xlabel('Confidence')
			plt.legend(loc='best', fancybox=True, framealpha=0.5)
			#~ gs1.tight_layout(fig,rect=(0.10, 0.70, 0.90, 0.95), pad=0, w_pad=0.03)
			logging.debug('Completed confidence axes plot, legend and labels')
			
			if logscale:
				vmin = np.min([np.min(subj['hit_histogram'][(subj['hit_histogram']>0).nonzero()]),
							   np.min(subj['miss_histogram'][(subj['miss_histogram']>0).nonzero()])])
				norm = LogNorm()
			else:
				vmin = np.min([np.min(subj['hit_histogram']),
							   np.min(subj['miss_histogram']),
							   np.min(model['hit_histogram']),
							   np.min(model['miss_histogram'])])
				norm = None
			vmax = np.max([np.max([subj['hit_histogram'],subj['miss_histogram']]),
						   np.max([model['hit_histogram'],model['miss_histogram']])])
			
			ax00 = plt.subplot(gs2[0,0])
			logging.debug('Created subject hit axes')
			plt.imshow(subj['hit_histogram'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
						extent=[subj['t_array'][0],subj['t_array'][-1],0,1],norm=norm)
			plt.ylabel('Confidence')
			plt.title('Hit')
			logging.debug('Populated subject hit axes')
			ax10 = plt.subplot(gs2[1,0],sharex=ax00,sharey=ax00)
			logging.debug('Created model hit axes')
			plt.imshow(model['hit_histogram'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
						extent=[model['t_array'][0],model['t_array'][-1],0,1],norm=norm)
			plt.xlabel('RT [{time_units}]'.format(time_units=self.time_units()))
			plt.ylabel('Confidence')
			logging.debug('Populated model hit axes')
			
			ax01 = plt.subplot(gs2[0,1],sharex=ax00,sharey=ax00)
			logging.debug('Created subject miss axes')
			plt.imshow(subj['miss_histogram'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
						extent=[subj['t_array'][0],subj['t_array'][-1],0,1],norm=norm)
			plt.title('Miss')
			logging.debug('Populated subject miss axes')
			ax11 = plt.subplot(gs2[1,1],sharex=ax00,sharey=ax00)
			logging.debug('Created model miss axes')
			im = plt.imshow(model['miss_histogram'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
						extent=[model['t_array'][0],model['t_array'][-1],0,1],norm=norm)
			plt.xlabel('RT [{time_units}]'.format(time_units=self.time_units()))
			if xlim_rt_cutoff:
				ax00.set_xlim([0,rt_cutoff])
			logging.debug('Populated model miss axes')
			
			ax00.tick_params(labelleft=True, labelbottom=False)
			ax01.tick_params(labelleft=False, labelbottom=False)
			ax10.tick_params(labelleft=True, labelbottom=True)
			ax11.tick_params(labelleft=False, labelbottom=True)
			logging.debug('Completed histogram axes')
			
			cbar_ax = plt.subplot(gs3[0])
			logging.debug('Created colorbar axes')
			plt.colorbar(im, cax=cbar_ax)
			plt.ylabel('Prob density')
			logging.debug('Completed colorbar axes')
			
			plt.suptitle(key)
			logging.debug('Sucessfully completed figure for key {0}'.format(key))
			if saver:
				logging.debug('Saving figure')
				if isinstance(saver,str):
					plt.savefig(saver,bbox_inches='tight')
				else:
					saver.savefig(fig,bbox_inches='tight')
			if show:
				logging.debug('Showing figure')
				plt.show(True)
	
	def save(self,fname):
		logging.debug('Fitter_plot_handler state that will be saved = "%s"',self.__getstate__())
		logging.info('Saving Fitter_plot_handler state to file "%s"',fname)
		f = open(fname,'w')
		pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
		f.close()
	
	def __setstate__(self,state):
		self.__init__(state['dictionary'],state['time_units'])
	
	def __getstate__(self):
		return {'dictionary':self.dictionary,'time_units':self._time_units}
	
	def merge(self,merge='all'):
		if merge=='subjects':
			key_aliaser = lambda key: re.sub('_subject_[\[\]\-0-9]+','',key)
		elif merge=='sessions':
			key_aliaser = lambda key: re.sub('_session_[\[\]\-0-9]+','',key)
		elif merge=='all':
			key_aliaser = lambda key: re.sub('_session_[\[\]\-0-9]+','',re.sub('_subject_[\[\]\-0-9]+','',key))
		else:
			raise ValueError('Unknown merge option={0}'.format(merge))
		output = Fitter_plot_handler({},self._time_units)
		return output.__aliased_iadd__(self,key_aliaser)

def parse_input():
	script_help = """ moving_bounds_fits.py help
 Sintax:
 moving_bounds_fits.py [option flag] [option value]
 
 moving_bounds_fits.py -h [or --help] displays help
 
 Optional arguments are:
 '-t' or '--task': Integer that identifies the task number when running multiple tasks
                   in parallel. By default it is one based but this behavior can be
                   changed with the option --task_base. [Default 1]
 '-nt' or '--ntasks': Integer that identifies the number tasks working in parallel [Default 1]
 '-tb' or '--task_base': Integer that identifies the task base. Can be 0 or 1, indicating
                         the task number of the root task. [Default 1]
 '-m' or '--method': String that identifies the fit method. Available values are full,
                     confidence_only, full_confidence, binary_confidence_only
                     and full_binary_confidence. [Default full]
 '-o' or '--optimizer': String that identifies the optimizer used for fitting.
                        Available values are 'cma' and all the scipy.optimize.minimize methods.
                        WARNING, cma is suited for problems with more than one dimensional
                        parameter spaces. If the optimization is performed on a single
                        dimension, the optimizer is changed to 'Nelder-Mead'. [Default cma]
 '-s' or '--save': This flag takes no values. If present it saves the figure.
 '--save_plot_handler': This flag takes no value. If present, the plot_handler is saved.
 '--load_plot_handler': This flag takes no value. If present, the plot_handler is loaded from the disk.
 '--show': This flag takes no values. If present it displays the plotted figure
           and freezes execution until the figure is closed.
 '--fit': This flag takes no values. If present it performs the fit for the selected
          method. By default, this flag is always set.
 '--no-fit': This flag takes no values. If present no fit is performed for the selected
             method. This flag should be used when it is only necesary to plot the results.
 '-u' or '--units': String that identifies the time units that will be used.
                    Available values are seconds and milliseconds. [Default seconds]
 '-sf' or '--suffix': A string suffix to paste to the filenames. [Default '']
 '--rt_cutoff': A Float that specifies the maximum RT in seconds to accept when
                loading subject data. Note that the "Luminancia" experiment
                forced subjects to respond in less than 1 second. Hence,
                an rt_cutoff greater than 1 second is supplied for the
                "Luminancia" experiment, it is chopped down to 1 second.
                [Defaults to 1 second for the Luminancia experiment
                and 14 seconds for the other experiments]
 '--plot_handler_rt_cutoff': Same as rt_cutoff but is used to create the
                             subject's RT histogram when constructing
                             the Fitter_plot_handler. [Default the fitter's rt_cutoff.
                             IMPORTANT! Fitter instances are saved with the rt_cutoff
                             value they were created with. If a fitter instance was
                             loaded from a file in order to get the Fitter_plot_handler,
                             then its rt_cutoff may be different than the value
                             specified in a separate run of the script.]
 '--confidence_partition': An Int that specifies the number of bins in which to partition
                           the [0,1] confidence report interval [Default 100]
 '--merge': Can be None, 'all', 'all_sessions' or 'all_subjects'. This parameter
            controls if and how the subject-session data should be merged before
            performing the fits. If merge is set to 'all', all the data is merged
            into a single "subjectSession". If merge is 'all_sessions', the
            data across all sessions for the same subject is merged together.
            If merge is 'all_subjects', the data across all subjects for a
            single session is merged. For all the above, the experiments are
            always treated separately. If merge is None, the data of every
            subject and session is treated separately. [Default None]
 '--plot_merge': Can be None, 'all', 'sessions' or 'subjects'. This parameter
            controls if and how the subject-session data should be merged when
            performing the plots. If merge is set to 'all', all subjects
            and sessions data are pooled together to plot figures that
            correspond to each experiment. If set to 'sessions' the sessions
            are pooled together and a separate figure is created for each
            subject and experiment. If set to 'subject', the subjects are
            pooled together and separate figures for each experiment and
            session are created. WARNING! This option does not override
            the --merge option. If the --merge option was set, the data
            that is used to perform the fits is merged together and
            cannot be divided again. --plot_merge allows the user to fit
            the data for every subject, experiment and session independently
            but to pool them together while plotting. [Default None]
 '-e' or '--experiment': Can be 'all', 'luminancia', '2afc' or 'auditivo'.
                         Indicates the experiment that you wish to fit. If set to
                         'all', all experiment data will be fitted. [Default 'all']
                         WARNING: is case insensitive.
 '-g' or '--debug': Activates the debug messages
 '-v' or '--verbose': Activates info messages (by default only warnings and errors
                      are printed).
 '-w': Override an existing saved fitter. This flag is only used if the 
       '--fit' flag is not disabled. If the flag '-w' is supplied, the script
       will override the saved fitter instance associated to the fitted
       subjectSession. If this flag is not supplied, the script will skip
       the subjectSession's that were already fitted.
 '-hcm' or '--high_confidence_mapping_method': Select the method with
       which the confidence mapping is computed. Two methods are available.
       1) 'log_odds': This mapping first takes the log_odds of the decision
          boundaries and then passes them through a sigmoid function that
          is affected by the 'high_confidence_threshold' and the
          'confidence_mapping_slope' parameters.
       2) 'belief': This mapping only applies a linear transformation to
          the belief bounds. The g bounds are first transformed as follows:
          the belief bound for the correct choice, gc, is transformed as
          gc = 2*gc-1. The belief bound for the incorrect choice, gi, is
          transformed as gi = 1-2*gi. These transformed bounds are then
          passed through the linear transformation:
          confidence_map_slope*(g - high_confidence_threshold). The values
          are then clipped to the interval [0,1], i.e. values greater than
          1 are converted to 1, and the values smaller than 0 are converted
          to 0.
       The default method is 'log_odds'. Be aware that, the mapping method
       is only added to the saved filename for the methods different than
       'log_odds'.
          
 
 The following argument values must be supplied as JSON encoded strings.
 JSON dictionaries are written as '{"key":val,"key2":val2}'
 JSON arrays (converted to python lists) are written as '[val1,val2,val3]'
 Note that the single quotation marks surrounding the brackets, and the
 double quotation marks surrounding the keys are mandatory. Furthermore,
 if a key value should be a string, it must also be enclosed in double
 quotes.
 
 '--fixed_parameters': A dictionary of fixed parameters. The dictionary must be written as
                       '{"fixed_parameter_name":fixed_parameter_value,...}'. For example,
                       '{"cost":0.2,"dead_time":0.5}'. Note that the value null
                       can be passed as a fixed parameter. In that case, the
                       parameter will be fixed to its default value or, if the flag
                       -f is also supplied, to the parameter value loaded from the
                       previous fitting round.
                       Default depends on the method. If the method is full, the
                       confidence parameters will be fixed, and in fact ignored.
                       If the method is confidence_only, the decision parameters
                       are fixed to their default values.
 
 '--start_point': A dictionary of starting points for the fitting procedure.
                  The dictionary must be written as '{"parameter_name":start_point_value,etc}'.
                  If a parameter is omitted, its default starting value is used. You only need to specify
                  the starting points for the parameters that you wish not to start at the default
                  start point. Default start points are estimated from the subjectSession data.
 
 '--bounds': A dictionary of lower and upper bounds in parameter space.
             The dictionary must be written as '{"parameter_name":[low_bound_value,up_bound_value],etc}'
             As for the --start_point option, if a parameter is omitted, its default bound is used.
             Default bounds are:
             '{"cost":[0.,10],"dead_time":[0.,0.4],"dead_time_sigma":[0.,3.],
               "phase_out_prob":[0.,1.],"internal_var":[self._internal_var*1e-6,self._internal_var*1e3],
               "high_confidence_threshold":[0.,3.],"confidence_map_slope":[0.,1e12]}'
 
 '--dpKwargs': A dictionary of optional keyword args used to construct a DecisionPolicy instance.
               Refer to DecisionPolicy in cost_time.py for posible key-value pairs. [Default '{}']
 
 '--optimizer_kwargs': A dictionary of options passed to the optimizer with a few additions.
                       If the optimizer is cma, refer to fmin in cma.py for the list of
                       posible cma options. The additional option in this case is only
                       'restarts':INTEGER that sets the number of restarts used in the cmaes fmin
                       function.
                       If a scipy optimizer is selected, refer to scipy minimize for
                       posible fmin options. The additional option in this case is
                       the 'repetitions':INTEGER that sets the number of independent
                       repetitions used by repeat_minize to find the minimum.
                       [Default depends on the optimizer. If 'cma', '{"restarts":1}'.
                       If not 'cma', '{"disp": False, "maxiter": 1000, "maxfev": 10000, "repetitions": 10}'
 
 '-f' or '--start_point_from_fit_output': A flag that tells the script to set the unspecified start_points
                       equal to the results of a previously saved fitting round. After the flag
                       the user must pass a dictionary of the form:
                       '{"method":"value","optimizer":"value","suffix":"value"}' where the values
                       must be the corresponding method, optimizer and suffix used by the
                       previous fitting round. The script will then try to load the fitted parameters
                       from the file:
                       fits_cognition/{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}{suffix}.pkl
                       where the experiment, name and session are taken from the subjectSession that
                       is currently being fitted, and the method, optimizer and suffix are the
                       values passed in the previously mentioned dictionary.
  
 Example:
 python moving_bounds_fits.py -t 1 -n 1 --save"""
	options =  {'task':1,'ntasks':1,'task_base':1,'method':'full','optimizer':'cma','save':False,
				'show':False,'fit':True,'time_units':'seconds','suffix':'','rt_cutoff':None,
				'merge':None,'fixed_parameters':{},'dpKwargs':{},'start_point':{},'bounds':{},
				'optimizer_kwargs':{},'experiment':'all','debug':False,'confidence_partition':100,
				'plot_merge':None,'verbose':False,'save_plot_handler':False,'load_plot_handler':False,
				'start_point_from_fit_output':None,'override':False,'plot_handler_rt_cutoff':None,
				'high_confidence_mapping_method':'log_odds'}
	expecting_key = True
	json_encoded_key = False
	key = None
	for i,arg in enumerate(sys.argv[1:]):
		if expecting_key:
			if arg=='-t' or arg=='--task':
				key = 'task'
				expecting_key = False
			elif arg=='-nt' or arg=='--ntasks':
				key = 'ntasks'
				expecting_key = False
			elif arg=='-tb' or arg=='--task_base':
				key = 'task_base'
				expecting_key = False
			elif arg=='-m' or arg=='--method':
				key = 'method'
				expecting_key = False
			elif arg=='-o' or arg=='--optimizer':
				key = 'optimizer'
				expecting_key = False
			elif arg=='-s' or arg=='--save':
				options['save'] = True
			elif arg=='--save_plot_handler':
				options['save_plot_handler'] = True
			elif arg=='--load_plot_handler':
				options['load_plot_handler'] = True
			elif arg=='--show':
				options['show'] = True
			elif arg=='-g' or arg=='--debug':
				options['debug'] = True
			elif arg=='-v' or arg=='--verbose':
				options['verbose'] = True
			elif arg=='--fit':
				options['fit'] = True
			elif arg=='--no-fit':
				options['fit'] = False
			elif arg=='-w':
				options['override'] = True
			elif arg=='-u' or arg=='--units':
				key = 'time_units'
				expecting_key = False
			elif arg=='--confidence_partition':
				key = 'confidence_partition'
				expecting_key = False
			elif arg=='-sf' or arg=='--suffix':
				key = 'suffix'
				expecting_key = False
			elif arg=='--rt_cutoff':
				key = 'rt_cutoff'
				expecting_key = False
			elif arg=='--plot_handler_rt_cutoff':
				key = 'plot_handler_rt_cutoff'
				expecting_key = False
			elif arg=='-e' or arg=='--experiment':
				key = 'experiment'
				expecting_key = False
			elif arg=='--fixed_parameters':
				key = 'fixed_parameters'
				expecting_key = False
				json_encoded_key = True
			elif arg=='--start_point':
				key = 'start_point'
				expecting_key = False
				json_encoded_key = True
			elif arg=='--bounds':
				key = 'bounds'
				expecting_key = False
				json_encoded_key = True
			elif arg=='--dpKwargs':
				key = 'dpKwargs'
				expecting_key = False
				json_encoded_key = True
			elif arg=='--optimizer_kwargs':
				key = 'optimizer_kwargs'
				expecting_key = False
				json_encoded_key = True
			elif arg=='-f' or arg=='--start_point_from_fit_output':
				key = 'start_point_from_fit_output'
				expecting_key = False
				json_encoded_key = True
			elif arg=='--plot_merge':
				key = 'plot_merge'
				expecting_key = False
			elif arg=='-hcm' or arg=='--high_confidence_mapping_method':
				key = 'high_confidence_mapping_method'
				expecting_key = False
			elif arg=='-h' or arg=='--help':
				print(script_help)
				sys.exit()
			else:
				raise RuntimeError("Unknown option: {opt} encountered in position {pos}. Refer to the help to see the list of options".format(opt=arg,pos=i+1))
		else:
			expecting_key = True
			if key in ['task','ntasks','task_base','confidence_partition']:
				options[key] = int(arg)
			elif key in ['rt_cutoff','plot_handler_rt_cutoff']:
				options[key] = float(arg)
			elif json_encoded_key:
				options[key] = json.loads(arg)
				json_encoded_key = False
			else:
				options[key] = arg
	if not expecting_key:
		raise RuntimeError("Expected a value after encountering key '{0}' but no value was supplied".format(arg))
	if options['task_base'] not in [0,1]:
		raise ValueError('task_base must be either 0 or 1')
	# Shift task from 1 base to 0 based if necessary
	options['task']-=options['task_base']
	if options['time_units'] not in ['seconds','milliseconds']:
		raise ValueError("Unknown supplied units: '{units}'. Available values are seconds and milliseconds".format(units=options['time_units']))
	if options['method'] not in ['full','confidence_only','full_confidence','binary_confidence_only','full_binary_confidence']:
		raise ValueError("Unknown supplied method: '{method}'. Available values are full, confidence_only full_confidence, binary_confidence_only and full_binary_confidence".format(method=options['method']))
	options['experiment'] = options['experiment'].lower()
	if options['experiment'] not in ['all','luminancia','2afc','auditivo']:
		raise ValueError("Unknown experiment supplied: '{0}'. Available values are 'all', 'luminancia', '2afc' and 'auditivo'".format(options['experiment']))
	else:
		# Switching case to the data_io_cognition case sensitive definition of each experiment
		options['experiment'] = {'all':'all','luminancia':'Luminancia','2afc':'2AFC','auditivo':'Auditivo'}[options['experiment']]
	
	if options['plot_merge'] not in [None,'subjects','sessions','all']:
		raise ValueError("Unknown plot_merge supplied: '{0}'. Available values are None, 'subjects', 'sessions' and 'all'.".format(options['plot_merge']))
	
	if not options['start_point_from_fit_output'] is None:
		keys = options['start_point_from_fit_output'].keys()
		if (not 'method' in keys) or (not 'optimizer' in keys) or (not 'suffix' in keys):
			raise ValueError("The supplied dictionary for 'start_point_from_fit_output' does not contain the all the required keys: 'method', 'optimizer' and 'suffix'")
	
	if options['high_confidence_mapping_method'].lower() not in ['log_odds','belief']:
		raise ValueError("The supplied high_confidence_mapping_method is unknown. Available values are 'log_odds' and 'belief'. Got {0} instead.".format(options['high_confidence_mapping_method']))
	else:
		options['high_confidence_mapping_method'] = options['high_confidence_mapping_method'].lower()
	
	return options

def prepare_fit_args(fitter,options,fname):
	temp = load_Fitter_from_file(fname)
	loaded_parameters = temp.get_parameters_dict_from_fit_output(temp._fit_output)
	for k in loaded_parameters.keys():
		if not k in temp.get_fitted_parameters():
			del loaded_parameters[k]
	logging.debug('Loaded parameters: {0}'.format(loaded_parameters))
	if temp.time_units!=fitter.time_units:
		logging.debug('Changing loaded parameters time units')
		for fp in loaded_parameters.keys():
			if fp in ['cost','internal_var']:
				scaling_factor = 1e3 if fitter.time_units=='seconds' else 1e-3
			elif fp in ['dead_time','dead_time_sigma']:
				scaling_factor = 1e-3 if fitter.time_units=='seconds' else 1e3
			else:
				scaling_factor = 1.
			loaded_parameters[fp]*=scaling_factor
	if fitter.method=='full':
		start_point = loaded_parameters.copy()
		fixed_parameters = loaded_parameters.copy()
		try:
			del fixed_parameters['cost']
		except KeyError:
			pass
		try:
			del fixed_parameters['dead_time']
		except KeyError:
			pass
		try:
			del fixed_parameters['dead_time_sigma']
		except KeyError:
			pass
		try:
			del fixed_parameters['internal_var']
		except KeyError:
			pass
		try:
			del fixed_parameters['phase_out_prob']
		except KeyError:
			pass
		try:
			del start_point['high_confidence_threshold']
		except KeyError:
			pass
		try:
			del start_point['confidence_map_slope']
		except KeyError:
			pass
	elif fitter.method=='confidence_only':
		fixed_parameters = loaded_parameters.copy()
		start_point = loaded_parameters.copy()
		try:
			del start_point['cost']
		except KeyError:
			pass
		try:
			del start_point['dead_time']
		except KeyError:
			pass
		try:
			del start_point['dead_time_sigma']
		except KeyError:
			pass
		try:
			del start_point['internal_var']
		except KeyError:
			pass
		try:
			del start_point['phase_out_prob']
		except KeyError:
			pass
		try:
			del fixed_parameters['high_confidence_threshold']
		except KeyError:
			pass
		try:
			del fixed_parameters['confidence_map_slope']
		except KeyError:
			pass
	elif fitter.method=='binary_confidence_only':
		fixed_parameters = loaded_parameters.copy()
		start_point = loaded_parameters.copy()
		try:
			del start_point['cost']
		except KeyError:
			pass
		try:
			del start_point['dead_time']
		except KeyError:
			pass
		try:
			del start_point['dead_time_sigma']
		except KeyError:
			pass
		try:
			del start_point['internal_var']
		except KeyError:
			pass
		try:
			del start_point['phase_out_prob']
		except KeyError:
			pass
		try:
			del fixed_parameters['high_confidence_threshold']
		except KeyError:
			pass
	elif fitter.method=='full_binary_confidence':
		start_point = loaded_parameters.copy()
		fixed_parameters = loaded_parameters.copy()
		try:
			del fixed_parameters['cost']
		except KeyError:
			pass
		try:
			del fixed_parameters['dead_time']
		except KeyError:
			pass
		try:
			del fixed_parameters['dead_time_sigma']
		except KeyError:
			pass
		try:
			del fixed_parameters['internal_var']
		except KeyError:
			pass
		try:
			del fixed_parameters['phase_out_prob']
		except KeyError:
			pass
		try:
			del start_point['high_confidence_threshold']
		except KeyError:
			pass
	else:
		start_point = loaded_parameters.copy()
		fixed_parameters = {}
	
	for k in options['fixed_parameters'].keys():
		if options['fixed_parameters'][k] is None:
			try:
				fixed_parameters[k] = start_point[k]
			except:
				fixed_parameters[k] = None
		else:
			fixed_parameters[k] = options['fixed_parameters'][k]
	start_point.update(options['start_point'])
	if fitter.method=='full_binary_confidence' or fitter.method=='binary_confidence_only':
		try:
			del start_point['confidence_map_slope']
		except KeyError:
			pass
		fixed_parameters['confidence_map_slope'] = np.inf
	logging.debug('Prepared fixed_parameters = {0}'.format(fixed_parameters))
	logging.debug('Prepared start_point = {0}'.format(start_point))
	return fixed_parameters,start_point

if __name__=="__main__":
	# Parse input from sys.argv
	options = parse_input()
	if options['debug']:
		logging.basicConfig(level=logging.DEBUG)
	elif options['verbose']:
		logging.basicConfig(level=logging.INFO)
	else:
		logging.basicConfig(level=logging.WARNING)
	task = options['task']
	ntasks = options['ntasks']
	
	# Prepare subjectSessions list
	subjects = io.filter_subjects_list(io.unique_subject_sessions(raw_data_dir),'all_sessions_by_experiment')
	if options['experiment']!='all':
		subjects = io.filter_subjects_list(subjects,'experiment_'+options['experiment'])
	logging.debug('Total number of subjectSessions listed = {0}'.format(len(subjects)))
	logging.debug('Total number of subjectSessions that will be fitted = {0}'.format(len(range(task,len(subjects),ntasks))))
	fitter_plot_handler = None
	
	# Main loop over subjectSessions
	for i,s in enumerate(subjects):
		logging.debug('Enumerated {0} subject {1}'.format(i,s.get_key()))
		if (i-task)%ntasks==0:
			logging.info('Task will execute for enumerated {0} subject {1}'.format(i,s.get_key()))
			# Fit parameters if the user did not disable the fit flag
			if options['fit']:
				logging.debug('Flag "fit" was True')
				fitter = Fitter(s,time_units=options['time_units'],method=options['method'],\
					   optimizer=options['optimizer'],decisionPolicyKwArgs=options['dpKwargs'],\
					   suffix=options['suffix'],rt_cutoff=options['rt_cutoff'],\
					   confidence_partition=options['confidence_partition'],\
					   high_confidence_mapping_method=options['high_confidence_mapping_method'])
				fname = fitter.get_save_file_name()
				if options['override'] or not (os.path.exists(fname) and os.path.isfile(fname)):
					# Set start point and fixed parameters to the user supplied values
					# Or to the parameters loaded from a previous fit round
					if options['start_point_from_fit_output']:
						logging.debug('Flag start_point_from_fit_output was present. Will load parameters from previous fit round')
						loaded_method = options['start_point_from_fit_output']['method']
						loaded_optimizer = options['start_point_from_fit_output']['optimizer']
						loaded_suffix = options['start_point_from_fit_output']['suffix']
						fname = Fitter_filename(experiment=s.experiment,method=loaded_method,name=s.get_name(),\
												session=s.get_session(),optimizer=loaded_optimizer,suffix=loaded_suffix,\
												confidence_map_method=options['high_confidence_mapping_method'])
						logging.debug('Will load parameters from file: {0}'.format(fname))
						fixed_parameters,start_point = prepare_fit_args(fitter,options,fname)
					else:
						fixed_parameters = options['fixed_parameters']
						start_point = options['start_point']
					bounds = options['bounds']
					
					# Perform fit and save fit output
					fit_output = fitter.fit(fixed_parameters=fixed_parameters,\
											start_point=start_point,\
											bounds=bounds,\
											optimizer_kwargs=options['optimizer_kwargs'])
					fitter.save()
				else:
					logging.warning('File {0} already exists, will skip enumerated subject {1} whose key is {2}. If you wish to override saved Fitter instances, supply the flag -w.'.format(fname,i,s.get_key()))
			# Prepare plotable data
			if options['show'] or options['save'] or options['save_plot_handler']:
				logging.debug('show, save or save_plot_fitter flags were True.')
				if options['load_plot_handler']:
					fname = Fitter_filename(experiment=s.experiment,method=options['method'],name=s.get_name(),
							session=s.get_session(),optimizer=options['optimizer'],suffix=options['suffix'],
							confidence_map_method=options['high_confidence_mapping_method']).replace('.pkl','_plot_handler.pkl')
					logging.debug('Loading Fitter_plot_handler from file={0}'.format(fname))
					try:
						f = open(fname,'r')
						temp = pickle.load(f)
						f.close()
					except:
						logging.warning('Failed to load Fitter_plot_handler from file={0}. Will continue to next subject.'.format(fname))
						continue
				else:
					fname = Fitter_filename(experiment=s.experiment,method=options['method'],name=s.get_name(),
							session=s.get_session(),optimizer=options['optimizer'],suffix=options['suffix'],
							confidence_map_method=options['high_confidence_mapping_method'])
					# Try to load the fitted data from file 'fname' or continue to next subject
					try:
						logging.debug('Attempting to load fitter from file "{0}".'.format(fname))
						fitter = load_Fitter_from_file(fname)
					except:
						logging.warning('Failed to load fitter from file {0}. Will continue to next subject.'.format(fname))
						continue
					# Create Fitter_plot_handler for the loaded Fitter instance
					logging.debug('Getting Fitter_plot_handler with merge_plot={0}.'.format(options['plot_merge']))
					if not options['plot_handler_rt_cutoff'] is None:
						if s.experiment=='Luminancia':
							cutoff = np.min([1.,options['plot_handler_rt_cutoff']])
						else:
							cutoff = options['plot_handler_rt_cutoff']
						logging.debug('Fitter_plot_handler will use rt_cutoff = {0}'.format(cutoff))
						edges = np.linspace(0,cutoff,51)
					else:
						edges = None
					temp = fitter.get_fitter_plot_handler(merge=options['plot_merge'],edges=edges)
					if options['save_plot_handler']:
						fname = fname.replace('.pkl','_plot_handler.pkl')
						if options['override'] or not (os.path.exists(fname) and os.path.isfile(fname)):
							logging.debug('Saving Fitter_plot_handler to file={0}.'.format(fname))
							temp.save(fname)
						else:
							logging.warning('Could not save Fitter_plot_handler. File {0} already exists. To override supply the flag -w.'.format(fname))
				# Add the new Fitter_plot_handler to the bucket of plot handlers
				logging.debug('Adding Fitter_plot_handlers')
				if fitter_plot_handler is None:
					fitter_plot_handler = temp
				else:
					fitter_plot_handler+= temp
	
	# Prepare figure saver
	if options['save']:
		if task==0 and ntasks==1:
			fname = "fits_cognition_{experiment}{method}_{cmapmeth}{suffix}".format(\
					experiment=options['experiment']+'_' if options['experiment']!='all' else '',\
					method=options['method'],suffix=options['suffix'],
					cmapmeth=options['high_confidence_mapping_method'])
		else:
			fname = "fits_cognition_{experiment}{method}_{cmapmeth}_{task}_{ntasks}{suffix}".format(\
					experiment=options['experiment']+'_' if options['experiment']!='all' else '',\
					method=options['method'],task=task,ntasks=ntasks,suffix=options['suffix'],\
					cmapmeth=options['high_confidence_mapping_method'])
		if os.path.isdir("../../figs"):
			fname = "../../figs/"+fname
		if loc==Location.cluster:
			fname+='.png'
			saver = fname
		else:
			fname+='.pdf'
			saver = PdfPages(fname)
	else:
		saver = None
	# Plot and show, or plot and save depending on the flags supplied by the user
	if options['show'] or options['save']:
		logging.debug('Plotting results from fitter_plot_handler')
		assert not fitter_plot_handler is None, 'Could not create the Fitter_plot_handler to plot the fitter results'
		if options['plot_merge'] and options['load_plot_handler']:
			fitter_plot_handler = fitter_plot_handler.merge(options['plot_merge'])
		fitter_plot_handler.plot(saver=saver,show=options['show'])
		if options['save']:
			logging.debug('Closing saver')
			saver.close()
