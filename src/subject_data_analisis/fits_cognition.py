from __future__ import division

import enum, os, sys, math, scipy, pickle, warnings, json, logging, copy
import numpy as np
from utils import normpdf

class Location(enum.Enum):
	facu = 0
	home = 1
	cluster = 2
	unknown = 3

opsys,computer_name,kern,bla,bits = os.uname()
if opsys.lower().startswith("linux"):
	if computer_name=="facultad":
		loc = Location.facu
	elif computer_name.startswith("sge"):
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
	import matplotlib as mt
	if loc==Location.cluster:
		mt.use('Agg')
	from matplotlib import pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	import matplotlib.gridspec as gridspec
	can_plot = True
except:
	can_plot = False
import data_io_cognition as io
import cost_time as ct
import cma

if np.__version__<'1.9':
	orig_unique = np.unique
	def np19_unique(ar, return_index=False, return_inverse=False, return_counts=False):
		"\n    Find the unique elements of an array.\n\n    Returns the sorted unique elements of an array. There are three optional\n    outputs in addition to the unique elements: the indices of the input array\n    that give the unique values, the indices of the unique array that\n    reconstruct the input array, and the number of times each unique value\n    comes up in the input array.\n\n    Parameters\n    ----------\n    ar : array_like\n        Input array. This will be flattened if it is not already 1-D.\n    return_index : bool, optional\n        If True, also return the indices of `ar` that result in the unique\n        array.\n    return_inverse : bool, optional\n        If True, also return the indices of the unique array that can be used\n        to reconstruct `ar`.\n    return_counts : bool, optional\n        .. versionadded:: 1.9.0\n        If True, also return the number of times each unique value comes up\n        in `ar`.\n\n    Returns\n    -------\n    unique : ndarray\n        The sorted unique values.\n    unique_indices : ndarray, optional\n        The indices of the first occurrences of the unique values in the\n        (flattened) original array. Only provided if `return_index` is True.\n    unique_inverse : ndarray, optional\n        The indices to reconstruct the (flattened) original array from the\n        unique array. Only provided if `return_inverse` is True.\n    unique_counts : ndarray, optional\n        .. versionadded:: 1.9.0\n        The number of times each of the unique values comes up in the\n        original array. Only provided if `return_counts` is True.\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Examples\n    --------\n    >>> np.unique([1, 1, 2, 2, 3, 3])\n    array([1, 2, 3])\n    >>> a = np.array([[1, 1], [2, 3]])\n    >>> np.unique(a)\n    array([1, 2, 3])\n\n    Return the indices of the original array that give the unique values:\n\n    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])\n    >>> u, indices = np.unique(a, return_index=True)\n    >>> u\n    array(['a', 'b', 'c'],\n           dtype='|S1')\n    >>> indices\n    array([0, 1, 3])\n    >>> a[indices]\n    array(['a', 'b', 'c'],\n           dtype='|S1')\n\n    Reconstruct the input array from the unique values:\n\n    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])\n    >>> u, indices = np.unique(a, return_inverse=True)\n    >>> u\n    array([1, 2, 3, 4, 6])\n    >>> indices\n    array([0, 1, 4, 3, 1, 2, 1])\n    >>> u[indices]\n    array([1, 2, 6, 4, 2, 3, 2])\n\n    "
		_return_inverse = return_inverse
		if return_counts:
			_return_inverse = True
		ret = orig_unique(ar,return_index=return_index,return_inverse=_return_inverse)
		if return_counts:
			if return_index:
					unique_inverse = ret[2]
			else:
				unique_inverse = ret[1]
			unique_counts = np.zeros_like(ret[0])
			for i,u in enumerate(ret[0]):
					unique_counts[i] = np.sum(ar==u)
			ret+=(unique_counts,)
		return ret
	np.unique = np19_unique

if np.__version__<'1.8':
	def np18_nanmean(a, axis=None, dtype=None, out=None):
		"Compute the arithmetic mean along the specified axis, ignoring NaNs.\n\nReturns the average of the array elements.  The average is taken over the flattened array by default, otherwise over the specified axis. `float64` intermediate and return values are used for integer inputs.\n\nFor all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.\n.. versionadded:: 1.8.0\n\nParameters\n----------\na : array_like\n    Array containing numbers whose mean is desired. If `a` is not an array, a conversion is attempted.\naxis : int, optional\n    Axis along which the means are computed. The default is to compute the mean of the flattened array.\ndtype : data-type, optional\n    Type to use in computing the mean.  For integer inputs, the default is `float64`; for inexact inputs, it is the same as the input dtype.\nout : ndarray, optional\n    Alternate output array in which to place the result.  The default is ``None``; if provided, it must have the same shape as the expected output, but the type will be cast if necessary.  See `doc.ufuncs` for details.\n\nReturns\n-------\nm : ndarray, see dtype parameter above\n    If `out=None`, returns a new array containing the mean values, otherwise a reference to the output array is returned. Nan is returned for slices that contain only NaNs.\n\nSee Also\n--------\naverage : Weighted averagemean : Arithmetic mean taken while not ignoring NaNs\nvar, nanvar\n\nNotes\n-----\nThe arithmetic mean is the sum of the non-NaN elements along the axis divided by the number of non-NaN elements.\n\nNote that for floating-point input, the mean is computed using the same precision the input has.  Depending on the input data, this can cause the results to be inaccurate, especially for `float32`.  Specifying a higher-precision accumulator using the `dtype` keyword can alleviate this issue.\n\nExamples\n--------\n>>> a = np.array([[1, np.nan], [3, 4]])\n>>> np.nanmean(a)\n2.6666666666666665\n>>> np.nanmean(a, axis=0)\narray([ 2.,  4.])\n>>> np.nanmean(a, axis=1)\narray([ 1.,  3.5])"
		nans = np.isnan(a)
		n = np.sum(np.logical_not(nans),axis=axis,dtype=dtype)
		b = np.empty_like(a)
		b[:] = a
		b[nans] = 0
		if out is None:
			out = np.sum(b,axis=axis,dtype=dtype)
		else:
			np.sum(b,axis=axis,dtype=dtype,out=out)
		out/=n
		return out
	np.nanmean = np18_nanmean

# Static functions
def add_dead_time(gs,dt,dead_time,dead_time_sigma,mode='full'):
	"""
	new_gs = add_dead_time(gs,dt,dead_time_sigma,mode='full')
	"""
	if dead_time_sigma<=0.:
		if dead_time_sigma==0:
			dead_time_shift = math.floor(dead_time/dt)
			output = np.zeros_like(gs)
			if dead_time_shift==0:
				output[:] = gs
			else:
				output[:,dead_time_shift:] = gs[:,:-dead_time_shift]
			return output
		else:
			raise ValueError("dead_time_sigma cannot take negative values. User supplied dead_time_sigma={0}".format(dead_time_sigma))
	gs = np.array(gs)
	conv_window = np.linspace(-dead_time_sigma*6,dead_time_sigma*6,int(math.ceil(dead_time_sigma*12/dt))+1)
	conv_window_size = conv_window.shape[0]
	conv_val = np.zeros_like(conv_window)
	conv_val[conv_window_size//2:] = normpdf(conv_window[conv_window_size//2:],0,dead_time_sigma)
	conv_val/=(np.sum(conv_val)*dt)
	cgs = []
	for g in gs:
		cgs.append(np.convolve(g,conv_val,mode=mode))
	cgs = np.array(cgs)
	a = int(0.5*(cgs.shape[1]-gs.shape[1]))
	if a==0:
		if cgs.shape[1]==gs.shape[1]:
			ret = cgs[:]
		else:
			ret = cgs[:,:-1]
	elif a==0.5*(cgs.shape[1]-gs.shape[1]):
		ret = cgs[:,a:-a]
	else:
		ret = cgs[:,a:-a-1]
	dead_time_shift = math.floor(dead_time/dt)
	output = np.zeros_like(ret)
	if dead_time_shift==0:
		output = ret
	else:
		output[:,dead_time_shift:] = ret[:,:-dead_time_shift]
	normalization = np.sum(output)*dt
	return tuple(output/normalization)

def rt_confidence_likelihood(confidence_matrix_time,confidence_matrix,RT,confidence):
	if RT>confidence_matrix_time[-1] or RT<confidence_matrix_time[0]:
		return 0.
	if confidence>1 or confidence<0:
		return 0.
	nC,nT = confidence_matrix.shape
	confidence_array = np.linspace(0,1,nC)
	t_ind = np.interp(RT,confidence_matrix_time,np.arange(0,nT,dtype=np.float))
	c_ind = np.interp(confidence,confidence_array,np.arange(0,nC,dtype=np.float))
	
	floor_t_ind = np.floor(t_ind)
	ceil_t_ind = np.ceil(t_ind)
	t_weight = 1.-t_ind%1.
	if floor_t_ind==nT-1:
		ceil_t_ind = floor_t_ind
		t_weight = np.array([1.])
	else:
		t_weight = np.array([1.-t_ind%1.,t_ind%1.])
	
	floor_c_ind = np.floor(c_ind)
	ceil_c_ind = np.ceil(c_ind)
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
	
	floor_t_ind = np.floor(t_ind)
	ceil_t_ind = np.ceil(t_ind)
	t_weight = 1.-t_ind%1.
	if floor_t_ind==nT-1:
		ceil_t_ind = floor_t_ind
		weight = np.array([1.])
	else:
		weight = np.array([1.-t_ind%1.,t_ind%1.])
	
	prob = np.sum(decision_pdf[floor_t_ind:ceil_t_ind+1]*weight)
	return prob

def load_Fitter_from_file(fname):
	f = open(fname,'r')
	fitter = pickle.load(f)
	f.close()
	return fitter

class Fitter:
	# Initer
	def __init__(self,subjectSession,time_units='seconds',method='full',optimizer='cma',\
				decisionPolicyKwArgs={},suffix='',rt_cutoff=None,confidence_partition=100):
		logging.info('Creating Fitter instance for "{experiment}" experiment and "{name}" subject with sessions={session}'.format(
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
		self.confidence_partition = float(int(confidence_partition))
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
		self.performance = dat[:,2]
		self.confidence = dat[:,3]
		logging.debug('Trials loaded = %d',len(self.performance))
		if self.experiment=='Luminancia':
			self.mu,self.mu_indeces,count = np.unique(self.contrast/self._ISI,return_inverse=True,return_counts=True)
		else:
			self.mu,self.mu_indeces,count = np.unique(self.contrast,return_inverse=True,return_counts=True)
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
			self.confidence_partition = state['confidence_partition']
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
	def get_parameters_dict(self,x):
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
		return self.fixed_parameters
	
	def get_fitted_parameters(self):
		return self.fitted_parameters
	
	def get_start_point(self):
		return self._start_point
	
	def get_bounds(self):
		return self.bounds
	
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
				 'raw_data_dir':self.raw_data_dir}
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
		
		_T = parameters['dead_time']+6*parameters['dead_time_sigma']
		_nT = int(_T/_dt)+1
		nT = int(_T/self.dp.dt)+1
		dense_conv_x = np.arange(0,_nT)*_dt
		if parameters['dead_time_sigma']>0:
			dense_conv_val = normpdf(dense_conv_x,parameters['dead_time'],parameters['dead_time_sigma'])
			dense_conv_val[dense_conv_x<parameters['dead_time']] = 0.
		else:
			dense_conv_val = np.zeros_like(dense_conv_x)
			dense_conv_val[np.floor(parameters['dead_time']/_dt)] = 1.
		conv_x = np.arange(0,nT)*self.dp.dt
		if must_downsample:
			ratio = np.ceil(_nT/nT)
			tail = _nT%nT
			if tail!=0:
				padded_cv = np.concatenate((dense_conv_val,np.nan*np.ones(nT-tail,dtype=np.float)),axis=0)
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
		subject = self._subjectSession_state['name']
		session = self._subjectSession_state['session']
		if isinstance(session,int):
			session = str(session)
		else:
			session = '-'.join([str(s) for s in session])
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
	
	def default_start_point(self):
		try:
			return self.__default_start_point__
		except:
			if hasattr(self,'fixed_parameters'):
				self.__default_start_point__ = self.fixed_parameters.copy()
			elif hasattr(self,'_fixed_parameters'):
				self.__default_start_point__ = self._fixed_parameters.copy()
			else:
				self.__default_start_point__ = {}
			if not 'cost' in self.__default_start_point__.keys():
				self.__default_start_point__['cost'] = 0.02 if self.time_units=='seconds' else 0.00002
			if not 'internal_var' in self.__default_start_point__.keys():
				self.__default_start_point__['internal_var'] = 1500. if self.time_units=='seconds' else 1.5
			if not 'phase_out_prob' in self.__default_start_point__.keys():
				self.__default_start_point__['phase_out_prob'] = 0.05
			if not 'dead_time' in self.__default_start_point__.keys():
				dead_time = sorted(self.rt)[int(0.025*len(self.rt))]
				self.__default_start_point__['dead_time'] = dead_time
			if not 'confidence_map_slope' in self.__default_start_point__.keys():
				self.__default_start_point__['confidence_map_slope'] = 17.2
			
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
				
				if not 'dead_time_sigma' in self.__default_start_point__.keys():
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
				if not 'high_confidence_threshold' in self.__default_start_point__.keys():
					self.__default_start_point__['high_confidence_threshold'] = log_odds[0][rt_mode_ind]
			else:
				if not 'high_confidence_threshold' in self.__default_start_point__.keys():
					self.__default_start_point__['high_confidence_threshold'] = 0.3
			
			return self.__default_start_point__
	
	def default_bounds(self):
		if self.time_units=='seconds':
			defaults = {'cost':[0.,10],'dead_time':[0.,1.5],'dead_time_sigma':[0.001,6.],
					'phase_out_prob':[0.,1.],'internal_var':[1.,1e5],
					'high_confidence_threshold':[0.,50.],'confidence_map_slope':[0.,1e3]}
		else:
			defaults = {'cost':[0.,0.01],'dead_time':[0.,1500.],'dead_time_sigma':[1.,6000.],
					'phase_out_prob':[0.,1.],'internal_var':[0.001,1e2],
					'high_confidence_threshold':[0.,50.],'confidence_map_slope':[0.,1e3]}
		default_sp = self.default_start_point()
		if default_sp['high_confidence_threshold']>defaults['high_confidence_threshold'][1]:
			defaults['high_confidence_threshold'][1] = 2*default_sp['high_confidence_threshold']
		return defaults
	
	def default_optimizer_kwargs(self):
		if self.optimizer=='cma':
			return {'restarts':1}
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
				merit_function = self.lum_full_merit
			elif self.method=='confidence_only':
				merit_function = self.lum_confidence_only_merit
			elif self.method=='full_confidence':
				merit_function = self.lum_full_confidence_merit
			else:
				raise ValueError('Unknown method "{0}" for experiment "{1}"'.format(self.method,self.experiment))
		elif self.experiment=='2AFC':
			if self.method=='full':
				merit_function = self.afc_full_merit
			elif self.method=='confidence_only':
				merit_function = self.afc_confidence_only_merit
			elif self.method=='full_confidence':
				merit_function = self.afc_full_confidence_merit
			else:
				raise ValueError('Unknown method "{0}" for experiment "{1}"'.format(self.method,self.experiment))
		elif self.experiment=='Auditivo':
			if self.method=='full':
				merit_function = self.aud_full_merit
			elif self.method=='confidence_only':
				merit_function = self.aud_confidence_only_merit
			elif self.method=='full_confidence':
				merit_function = self.aud_full_confidence_merit
			else:
				raise ValueError('Unknown method "{0}" for experiment "{1}"'.format(self.method,self.experiment))
		else:
			raise ValueError('Unknown experiment "{0}"'.format(self.experiment))
		self._fit_output = minimizer(merit_function)
		self.__fit_internals__ = None
		return self._fit_output
	
	# Savers
	def save(self):
		logging.debug('Fitter state that will be saved = "%s"',self.__getstate__())
		if not hasattr(self,'_fit_output'):
			raise ValueError('The Fitter instance has not performed any fit and still has no _fit_output attribute set')
		session = self._subjectSession_state['session']
		if isinstance(session,int):
			session = str(session)
		else:
			session = '-'.join([str(s) for s in session])
		fname = 'fits_cognition/{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}{suffix}.pkl'.format(
				experiment=self.experiment,method=self.method,name=self._subjectSession_state['name'],
				session=session,optimizer=self.optimizer,suffix=self.suffix)
		logging.info('Saving Fitter state to file "%s"',fname)
		f = open(fname,'w')
		pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
		f.close()
	
	# Sanitizers
	def sanitize_parameters_x0_bounds(self):
		fittable_parameters = self.get_fittable_parameters()
		confidence_parameters = self.get_confidence_parameters()
		method_fitted_parameters = {'full_confidence':self._fitted_parameters[:],'full':[],'confidence_only':[]}
		method_fixed_parameters = {'full_confidence':self._fixed_parameters.copy(),'full':self._fixed_parameters.copy(),'confidence_only':self._fixed_parameters.copy()}
		method_sp = {'full_confidence':[],'full':[],'confidence_only':[]}
		method_b = {'full_confidence':[],'full':[],'confidence_only':[]}
		for par in self._fitted_parameters:
			if par not in confidence_parameters:
				method_fitted_parameters['full'].append(par)
				method_sp['full'].append(self._start_point[par])
				method_b['full'].append(self._bounds[par])
				if par not in method_fixed_parameters['confidence_only'].keys():
					method_fixed_parameters['confidence_only'][par] = self._start_point[par]
			else:
				method_fitted_parameters['confidence_only'].append(par)
				method_sp['confidence_only'].append(self._start_point[par])
				method_b['confidence_only'].append(self._bounds[par])
				if par not in method_fixed_parameters['full'].keys():
					method_fixed_parameters['full'][par] = self._start_point[par]
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
		logging.info('Sanitizing minizer output')
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
		logging.info('Initing minimizer')
		logging.debug('init_minimizer args: start_point=%(start_point)s, bounds=%(bounds)s, optimizer_kwargs=%(optimizer_kwargs)s',{'start_point':start_point,'bounds':bounds,'optimizer_kwargs':optimizer_kwargs})
		if self.optimizer=='cma':
			scaling_factor = bounds[1]-bounds[0]
			logging.debug('scaling_factor = %s',scaling_factor)
			options = {'bounds':bounds,'CMA_stds':scaling_factor}
			options.update(optimizer_kwargs)
			restarts = options['restarts']
			del options['restarts']
			options = cma.CMAOptions(options)
			minimizer = lambda x: self.sanitize_fmin_output(cma.fmin(x,start_point,1./3.,options,restarts=restarts),package='cma')
			#~ minimizer = lambda x: self.sanitize_fmin_output((start_point,None,None,None,None,None,None,None),'cma')
		else:
			repetitions = optimizer_kwargs['repetitions']
			_start_points = [start_point]
			for rsp in np.random.rand(repetitions-1,len(start_point)):
				temp = []
				for val,(lb,ub) in zip(rsp,bounds):
					temp.append(val*(ub-lb)+lb)
				_start_points.append(np.array(temp))
			start_point_generator = iter(_start_points)
			minimizer = lambda x: self.sanitize_fmin_output(self.repeat_minimize(x,start_point_generator,bounds=bounds,optimizer_kwargs=optimizer_kwargs),package='scipy')
			#~ minimizer = lambda x: self.sanitize_fmin_output({'xbest':start_point,'funbest':None,'nfev':None,'nit':None,'xmean':None,'xstd':None},'scipy')
		
		return minimizer
	
	def repeat_minimize(self,merit,start_point_generator,bounds,optimizer_kwargs):
		output = {'xs':[],'funs':[],'nfev':0,'nit':0,'xbest':None,'funbest':None,'xmean':None,'xstd':None,'funmean':None,'funstd':None}
		repetitions = 0
		for start_point in start_point_generator:
			repetitions+=1
			logging.info('Round {2} with start_point={0} and bounds={1}'.format(start_point, bounds,repetitions))
			res = scipy.optimize.minimize(merit,start_point, method=optimizer,bounds=bounds,options=optimizer_kwargs)
			logging.info('New round with start_point={0} and bounds={0}'.format(start_point, bounds))
			logging.info('Round {0} ended. Fun val: {1}. x={2}'.format(repetitions,res.fun,res.x))
			output['xs'].append(res.x)
			output['funs'].append(res.fun)
			output['nfev']+=res.nfev
			output['nit']+=res.nit
			if output['funbest'] is None or res.fun<output['funbest']:
				output['funbest'] = res.fun
				output['xbest'] = res.x
			logging.info('Best so far: {0} at point {1}'.format(output['funbest'],output['xbest']))
		arr_xs = np.array(output['xs'])
		arr_funs = np.array(output['funs'])
		output['xmean'] = np.mean(arr_xs)
		output['xstd'] = np.std(arr_xs)
		output['funmean'] = np.mean(arr_funs)
		output['funstd'] = np.std(arr_funs)
		return output
	
	# Auxiliary method
	def high_confidence_mapping(self,high_confidence_threshold,confidence_map_slope):
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
		# These issues are resolved naturally in the two line statements
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			phigh = 1./(1.+np.exp(confidence_map_slope*(high_confidence_threshold-log_odds)))
		phigh[high_confidence_threshold==log_odds] = 0.5
		
		if _dt:
			ratio = np.ceil(_nT/self.dp.nT)
			tail = _nT%self.dp.nT
			if tail!=0:
				padded_phigh = np.concatenate((phigh,np.nan*np.ones((2,self.dp.nT-tail),dtype=np.float)),axis=1)
			else:
				padded_phigh = phigh
			padded_phigh = np.reshape(padded_phigh,(2,-1,ratio))
			phigh = np.nanmean(padded_phigh,axis=2)
		return phigh
	
	def confidence_mapping_pdf_matrix(self,first_passage_pdfs,parameters,mapped_confidences=None,return_unconvoluted_matrix=False):
		indeces = np.arange(0,self.confidence_partition,dtype=np.float)
		confidence_array = np.linspace(0,1,self.confidence_partition)
		nT = self.dp.nT
		confidence_matrix = np.zeros((2,self.confidence_partition,nT))
		if mapped_confidences is None:
			mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		conv_val,conv_x = self.get_dead_time_convolver(parameters)
		_nT = len(conv_x)
		conv_confidence_matrix = np.zeros((2,self.confidence_partition,nT+_nT))
		for decision_ind,(first_passage_pdf,mapped_confidence) in enumerate(zip(first_passage_pdfs,mapped_confidences)):
			cv_inds = np.interp(mapped_confidence,confidence_array,indeces)
			for index,(cv_ind,floor_cv_ind,ceil_cv_ind,fppdf) in enumerate(zip(cv_inds,np.floor(cv_inds),np.ceil(cv_inds),first_passage_pdf)):
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
							
							prev_temp_fppdf = np.polyval(prev_polypars,np.arange(prior_ceil_cv_ind+1,ceil_cv_ind+1))
							curr_temp_fppdf = np.polyval(curr_polypars,np.arange(prior_ceil_cv_ind+1,ceil_cv_ind+1))
							prev_temp_fppdf[prev_temp_fppdf<0.] = 0.
							curr_temp_fppdf[curr_temp_fppdf<0.] = 0.
							
							prev_norm+= np.sum(prev_temp_fppdf)
							norm = np.sum(curr_temp_fppdf)
							confidence_matrix[decision_ind,prior_ceil_cv_ind+1:ceil_cv_ind+1,index-1]+= prev_temp_fppdf
							confidence_matrix[decision_ind,prior_ceil_cv_ind+1:ceil_cv_ind+1,index] = curr_temp_fppdf
						else:
							prev_polypars = np.polyfit([prior_cv_ind,floor_cv_ind],[prior_fppdf,0.],1)
							curr_polypars = np.polyfit([prior_cv_ind,floor_cv_ind],[0.,fppdf],1)
							
							prev_temp_fppdf = np.polyval(prev_polypars,np.arange(prior_floor_cv_ind-1,floor_cv_ind-1,-1))
							curr_temp_fppdf = np.polyval(curr_polypars,np.arange(prior_floor_cv_ind-1,floor_cv_ind-1,-1))
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
					if index<len(cv_inds)-1:
						end_index = _nT+index-1
						cv_end_index = end_index-index+1
						conv_confidence_matrix[decision_ind,:,index-1:end_index]+= np.reshape(confidence_matrix[decision_ind,:,index-1],(-1,1))*conv_val[:cv_end_index]
					else:
						if norm>0:
							confidence_matrix[decision_ind,:,index]*=fppdf/norm
						end_index = _nT+index
						cv_end_index = end_index-index
						conv_confidence_matrix[decision_ind,:,index:end_index]+= np.reshape(confidence_matrix[decision_ind,:,index],(-1,1))*conv_val[:cv_end_index]
					prev_norm = norm
				prior_fppdf = fppdf
				prior_cv_ind = cv_ind
				prior_floor_cv_ind = floor_cv_ind
				prior_ceil_cv_ind = ceil_cv_ind
		conv_confidence_matrix/=(np.sum(conv_confidence_matrix)*self.dp.dt)
		if return_unconvoluted_matrix:
			confidence_matrix/=(np.sum(confidence_matrix)*self.dp.dt)
			output = (conv_confidence_matrix,confidence_matrix)
		else:
			output = conv_confidence_matrix
		return output
	
	def decision_pdf(self,first_passage_pdfs,parameters):
		conv_val,conv_x = self.get_dead_time_convolver(parameters)
		_nT = len(conv_x)
		decision_pdfs = np.zeros((2,self.dp.nT+_nT))
		for decision_ind,first_passage_pdf in enumerate(first_passage_pdfs):
			for index in range(self.dp.nT-1,-1,-1):
				decision_pdfs[decision_ind,index:index+_nT]+= first_passage_pdf[index]*conv_val
		return decision_pdfs
	
	def mapped_confidence_probability(self,confidence_mapping_pdf_matrix):
		return np.sum(confidence_mapping_pdf_matrix,axis=2)*self.dp.dt
	
	# Experiment dependant merits
	# Luminancia experiment
	def lum_full_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		for index,drift in enumerate(self.mu):
			gs = self.decision_pdf(np.array(self.dp.rt(drift,bounds=(xub,xlb))),parameters)
			t = np.arange(0,gs.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
				nlog_likelihood-= np.log(rt_likelihood(t,gs[1-int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def lum_confidence_only_merit(self,x):
		parameters = self.get_parameters_dict(x)
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
		random_rt_likelihood = parameters['phase_out_prob']/(self.max_RT-self.min_RT)/self.confidence_partition
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		for index,drift in enumerate(self.mu):
			if must_compute_first_passage_time:
				gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
				self.__fit_internals__['first_passage_times'][drift] = gs
			else:
				gs = self.__fit_internals__['first_passage_times'][drift]
			confidence_likelihood = np.sum(self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences),axis=0)
			t = np.arange(0,confidence_likelihood.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,conf in zip(self.rt[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,confidence_likelihood,rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def lum_full_confidence_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)/self.confidence_partition
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			confidence_likelihood = self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences)
			t = np.arange(0,confidence_likelihood.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,confidence_likelihood[1-int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	# 2AFC experiment
	def afc_full_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		for index,drift in enumerate(self.mu):
			gs = self.decision_pdf(np.array(self.dp.rt(drift,bounds=(xub,xlb))),parameters)
			t = np.arange(0,gs.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
				nlog_likelihood-= np.log(rt_likelihood(t,gs[1-int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def afc_confidence_only_merit(self,x):
		parameters = self.get_parameters_dict(x)
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
		random_rt_likelihood = parameters['phase_out_prob']/(self.max_RT-self.min_RT)/self.confidence_partition
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		for index,drift in enumerate(self.mu):
			if must_compute_first_passage_time:
				gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
				self.__fit_internals__['first_passage_times'][drift] = gs
			else:
				gs = self.__fit_internals__['first_passage_times'][drift]
			confidence_likelihood = np.sum(self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences),axis=0)
			t = np.arange(0,confidence_likelihood.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,conf in zip(self.rt[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,confidence_likelihood,rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def afc_full_confidence_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)/self.confidence_partition
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			confidence_likelihood = self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences)
			t = np.arange(0,confidence_likelihood.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,confidence_likelihood[1-int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	# Auditivo experiment
	def aud_full_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		for index,drift in enumerate(self.mu):
			gs = self.decision_pdf(np.array(self.dp.rt(drift,bounds=(xub,xlb))),parameters)
			t = np.arange(0,gs.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
				nlog_likelihood-= np.log(rt_likelihood(t,gs[1-int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def aud_confidence_only_merit(self,x):
		parameters = self.get_parameters_dict(x)
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
		random_rt_likelihood = parameters['phase_out_prob']/(self.max_RT-self.min_RT)/self.confidence_partition
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		for index,drift in enumerate(self.mu):
			if must_compute_first_passage_time:
				gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
				self.__fit_internals__['first_passage_times'][drift] = gs
			else:
				gs = self.__fit_internals__['first_passage_times'][drift]
			confidence_likelihood = np.sum(self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences),axis=0)
			t = np.arange(0,confidence_likelihood.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,conf in zip(self.rt[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,confidence_likelihood,rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def aud_full_confidence_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)/self.confidence_partition
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			confidence_likelihood = self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences)
			t = np.arange(0,confidence_likelihood.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,confidence_likelihood[1-int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	# Theoretical predictions
	def theoretical_rt_confidence_distribution(self,fit_output=None):
		parameters = self.get_parameters_dict_from_fit_output(fit_output)
		
		self.dp.set_cost(parameters['cost'])
		if self.experiment!='Luminancia':
			self.dp.set_internal_var(parameters['internal_var'])
		xub,xlb = self.dp.xbounds()
		
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)/self.confidence_partition
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		output = None
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			confidence_likelihood = self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences)
			
			if output is None:
				output = confidence_likelihood*self.mu_prob[index]
			else:
				output+= confidence_likelihood*self.mu_prob[index]
		output/=(np.sum(output)*self.dp.dt)
		t = np.arange(0,output.shape[2],dtype=np.float)*self.dp.dt
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)/self.confidence_partition*np.ones_like(t)
		random_rt_likelihood[np.logical_or(t<self.min_RT,t>self.max_RT)] = 0.
		return output*(1.-parameters['phase_out_prob'])+random_rt_likelihood, np.arange(0,output.shape[2],dtype=np.float)*self.dp.dt
	
	# Plotter
	def plot_fit(self,fit_output=None,saver=None,display=True):
		if not can_plot:
			raise ImportError('Could not import matplotlib package and it is imposible to plot fit')
		
		self.get_fitter_plot_handler(fit_output=fit_output).plot(saver=saver,display=display)

class Fitter_plot_handler():
	def __init__(self,obj,time_units):
		self._time_units = time_units
		self.required_data = ['hit_histogram','miss_histogram','rt','confidence']
		self.categories = ['experimental','theoretical']
		self.required_data_entries = {'hit_histogram':['x','y','z'],'miss_histogram':['x','y','z'],'rt':['x','y'],'confidence':['x','y']}
		self.dictionary = {}
		try:
			for key in obj.keys():
				self.dictionary[key] = {}
				for category in self.categories:
					self.dictionary[key][category] = {}
					for required_data in self.required_data:
						self.dictionary[key][category][required_data] = {}
						for required_data_entry in self.required_data_entries[required_data]:
							self.dictionary[key][category][required_data][required_data_entry] = \
								copy.deepcopy(obj[key][category][required_data][required_data_entry])
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
	
	def __iadd__(self,other):
		for other_key in other.keys():
			if other_key in self.keys():
				for category in self.categories:
					for required_data in self.required_data:
						if required_data in ['hit_histogram','miss_histogram']:
							xorig = self[other_key][category][required_data]['x']
							xadded = other[other_key][category][required_data]['x']
							orig = self[other_key][category][required_data]['z']
							added = other[other_key][category][required_data]['z']
							if len(xorig)<len(xadded):
								self[other_key][category][required_data]['x'] = copy.copy(xadded)
								summed = added+np.pad(orig, ((0,0),(0,len(xadded)-len(xorig))),'constant',constant_values=(0., 0.))
								self[other_key][category][required_data]['z'] = summed
							elif len(xorig)>len(xadded):
								summed = orig+np.pad(added, ((0,0),(0,len(xorig)-len(xadded))),'constant',constant_values=(0., 0.))
								self[other_key][category][required_data]['z'] = summed
							else:
								self[other_key][category][required_data]['z']+= added
						elif required_data in ['rt']:
							xorig = self[other_key][category][required_data]['x']
							xadded = other[other_key][category][required_data]['x']
							orig = self[other_key][category][required_data]['y']
							added = other[other_key][category][required_data]['y']
							if len(xorig)<len(xadded):
								self[other_key][category][required_data]['x'] = copy.copy(xadded)
								summed = added+np.pad(orig, ((0,0),(0,len(xadded)-len(xorig))),'constant',constant_values=(0., 0.))
								self[other_key][category][required_data]['y'] = summed
							elif len(xorig)>len(xadded):
								summed = orig+np.pad(added, ((0,0),(0,len(xorig)-len(xadded))),'constant',constant_values=(0., 0.))
								self[other_key][category][required_data]['y'] = summed
							else:
								self[other_key][category][required_data]['y']+= added
						else:
							self[other_key][category][required_data]['y']+= \
								other[other_key][category][required_data]['y']
			else:
				self[other_key] = {}
				for category in self.categories:
					self[other_key][category] = {}
					for required_data in self.required_data:
						self[other_key][category][required_data] = {}
						for required_data_entry in self.required_data_entries[required_data]:
							self[other_key][category][required_data][required_data_entry] = \
								copy.deepcopy(other[other_key][category][required_data][required_data_entry])
		return self
	
	def __add__(self,other):
		output = Fitter_plot_handler(self)
		output+=other
		return output
	
	def normalize(self):
		for key in self.keys():
			for category in self.categories:
				for required_data in self.required_data:
					if required_data in ['hit_histogram','miss_histogram']:
						data = self[key][category][required_data]['z']
						t = self[key][category][required_data]['x']
						dt = t[1]-t[0]
						self[key][category][required_data]['z']/=(np.sum(data)*dt)
					elif required_data=='rt':
						data = self[key][category][required_data]['y']
						t = self[key][category][required_data]['x']
						dt = t[1]-t[0]
						self[key][category][required_data]['y']/=(np.sum(data)*dt)
					else:
						self[key][category][required_data]['y']/=np.sum(self[key][category][required_data]['y'])
	
	def plot(self,saver=None,display=True,xlim_rt_cutoff=True):
		if not can_plot:
			raise ImportError('Could not import matplotlib package and it is imposible to produce any plot')
		self.normalize()
		
		for key in self.keys():
			subj = self.dictionary[key]['experimental']
			model = self.dictionary[key]['theoretical']
			
			rt_cutoff = subj['rt']['x'][-1]+0.5*(subj['rt']['x'][-1]-subj['rt']['x'][-2])
			
			fig = plt.figure(figsize=(10,12))
			gs1 = gridspec.GridSpec(1, 2,left=0.10, right=0.90, wspace=0.1, top=0.95,bottom=0.70)
			gs2 = gridspec.GridSpec(2, 2,left=0.10, right=0.85, wspace=0.05, hspace=0.05, top=0.62,bottom=0.05)
			gs3 = gridspec.GridSpec(1, 1,left=0.87, right=0.90, wspace=0.1, top=0.62,bottom=0.05)
			axrt = plt.subplot(gs1[0])
			plt.step(subj['rt']['x'],subj['rt']['y'][0],'b',label='Subject hit')
			plt.step(subj['rt']['x'],-subj['rt']['y'][1],'r',label='Subject miss')
			plt.plot(model['rt']['x'],model['rt']['y'][0],'b',label='Model hit',linewidth=3)
			plt.plot(model['rt']['x'],-model['rt']['y'][1],'r',label='Model miss',linewidth=3)
			if xlim_rt_cutoff:
				axrt.set_xlim([0,rt_cutoff])
			plt.xlabel('RT [{time_units}]'.format(time_units=self.time_units()))
			plt.ylabel('Prob density')
			plt.legend(loc='best', fancybox=True, framealpha=0.5)
			axconf = plt.subplot(gs1[1])
			plt.step(subj['confidence']['x'],subj['confidence']['y'][0],'b',label='Subject hit')
			plt.step(subj['confidence']['x'],-subj['confidence']['y'][1],'r',label='Subject miss')
			plt.plot(model['confidence']['x'],model['confidence']['y'][0],'b',label='Model hit',linewidth=3)
			plt.plot(model['confidence']['x'],-model['confidence']['y'][1],'r',label='Model miss',linewidth=3)
			plt.xlabel('Confidence')
			plt.legend(loc='best', fancybox=True, framealpha=0.5)
			
			vmin = np.min([np.min([subj['hit_histogram']['z'],subj['miss_histogram']['z']]),
						   np.min([model['hit_histogram']['z'],model['miss_histogram']['z']])])
			vmax = np.max([np.max([subj['hit_histogram']['z'],subj['miss_histogram']['z']]),
						   np.max([model['hit_histogram']['z'],model['miss_histogram']['z']])])
			
			ax00 = plt.subplot(gs2[0,0])
			plt.imshow(subj['hit_histogram']['z'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
						extent=[subj['hit_histogram']['x'][0],subj['hit_histogram']['x'][-1],0,1])
			plt.ylabel('Confidence')
			plt.title('Hit')
			ax10 = plt.subplot(gs2[1,0],sharex=ax00,sharey=ax00)
			plt.imshow(model['hit_histogram']['z'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
						extent=[model['hit_histogram']['x'][0],model['hit_histogram']['x'][-1],0,1])
			plt.xlabel('RT [{time_units}]'.format(time_units=self.time_units()))
			plt.ylabel('Confidence')
			
			ax01 = plt.subplot(gs2[0,1],sharex=ax00,sharey=ax00)
			plt.imshow(subj['miss_histogram']['z'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
						extent=[subj['miss_histogram']['x'][0],subj['miss_histogram']['x'][-1],0,1])
			plt.title('Miss')
			ax11 = plt.subplot(gs2[1,1],sharex=ax00,sharey=ax00)
			im = plt.imshow(model['miss_histogram']['z'],aspect="auto",interpolation='none',origin='lower',vmin=vmin,vmax=vmax,
						extent=[model['miss_histogram']['x'][0],model['miss_histogram']['x'][-1],0,1])
			plt.xlabel('RT [{time_units}]'.format(time_units=self.time_units()))
			if xlim_rt_cutoff:
				ax00.set_xlim([0,rt_cutoff])
			
			ax00.tick_params(labelleft=True, labelbottom=False)
			ax01.tick_params(labelleft=False, labelbottom=False)
			ax10.tick_params(labelleft=True, labelbottom=True)
			ax11.tick_params(labelleft=False, labelbottom=True)
			
			cbar_ax = plt.subplot(gs3[0])
			plt.colorbar(im, cax=cbar_ax)
			plt.ylabel('Prob density')
			
			plt.suptitle(key)
			if saver:
				if isinstance(saver,str):
					plt.savefig(saver,bbox_inches='tight')
				else:
					saver.savefig(fig)
				if not display:
					plt.close(fig)
			if display:
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
                     confidence_only and full_confidence. [Default full]
 '-o' or '--optimizer': String that identifies the optimizer used for fitting.
                        Available values are 'cma' and all the scipy.optimize.minimize methods.
                        WARNING, cma is suited for problems with more than one dimensional
                        parameter spaces. If the optimization is performed on a single
                        dimension, the optimizer is changed to 'Nelder-Mead'. [Default cma]
 '-s' or '--save': This flag takes no values. If present it saves the figure.
 '--save_plot_handler': This flag takes no value. If present, the plot_handler is saved.
 '--load_plot_handler': This flag takes no value. If present, the plot_handler is loaded from the disk.
 '--plot': This flag takes no values. If present it displays the plotted figure
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
 '--plot_merge': Can be None, 'all', 'all_sessions' or 'all_subjects'. This parameter
            controls if and how the subject-session data should be merged before
            performing the fits. If merge is set to 'all', all the data is merged
            into a single "subjectSession". If merge is 'all_sessions', the
            data across all sessions for the same subject is merged together.
            If merge is 'all_subjects', the data across all subjects for a
            single session is merged. For all the above, the experiments are
            always treated separately. If merge is None, the data of every
            subject and session is treated separately. [Default None]
 '-e' or '--experiment': Can be 'all', 'luminancia', '2afc' or 'auditivo'.
                         Indicates the experiment that you wish to fit. If set to
                         'all', all experiment data will be fitted. [Default 'all']
                         WARNING: is case insensitive.
 '-g' or '--debug': Activates the debug messages
 '-v' or '--verbose': Activates info messages (by default only warnings and errors
                      are printed).
 
 The following argument values must be supplied as JSON encoded strings.
 JSON dictionaries are written as '{"key":val,"key2":val2}'
 JSON arrays (converted to python lists) are written as '[val1,val2,val3]'
 
 '--fixed_parameters': A dictionary of fixed parameters. The dictionary must be written as
                       '{"fixed_parameter_name":fixed_parameter_value,...}'. For example,
                       '{"cost":0.2,"dead_time":0.5}'. [Default depends on experiment.
                       For Luminance experiment '{"internal_var":0}'. For the other
                       experiments '{}'.]
 
 '--start_point': A dictionary of starting points for the fitting procedure.
                  The dictionary must be written as '{"parameter_name":start_point_value,etc}'.
                  If a parameter is omitted, its default starting value is used. You only need to specify
                  the starting points for the parameters that you wish not to start at the default
                  start point. Default start points are:
                  '{"cost":0.2 Hz,"dead_time":0.1 seconds,"dead_time_sigma":0.5 seconds,
                    "phase_out_prob":0.1,"internal_var":self._internal_var,
                    "high_confidence_threshold":0.5,"confidence_map_slope":1e9}'
                   The internal variance depends on the fitted experiment.
 
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
 
 Example:
 python moving_bounds_fits.py -t 1 -n 1 --save"""
	options =  {'task':1,'ntasks':1,'task_base':1,'method':'full','optimizer':'cma','save':False,
				'plot':False,'fit':True,'time_units':'seconds','suffix':'','rt_cutoff':None,
				'merge':None,'fixed_parameters':{},'dpKwargs':{},'start_point':{},'bounds':{},
				'optimizer_kwargs':{},'experiment':'all','debug':False,'confidence_partition':100,
				'plot_merge':None,'verbose':False,'save_plot_handler':False,'load_plot_handler':False}
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
			elif arg=='--plot':
				options['plot'] = True
			elif arg=='-g' or arg=='--debug':
				options['debug'] = True
			elif arg=='-v' or arg=='--verbose':
				options['verbose'] = True
			elif arg=='--fit':
				options['fit'] = True
			elif arg=='--no-fit':
				options['fit'] = False
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
			elif arg=='--plot_merge':
				key = 'plot_merge'
				expecting_key = False
			elif arg=='-h' or arg=='--help':
				print script_help
				sys.exit()
			else:
				raise RuntimeError("Unknown option: {opt} encountered in position {pos}. Refer to the help to see the list of options".format(opt=arg,pos=i+1))
		else:
			expecting_key = True
			if key in ['task','ntasks','task_base','confidence_partition']:
				options[key] = int(arg)
			elif json_encoded_key:
				option[key] = json.loads(arg)
				json_encoded_key = False
			else:
				options[key] = arg
	if options['task_base'] not in [0,1]:
		raise ValueError('task_base must be either 0 or 1')
	# Shift task from 1 base to 0 based if necessary
	options['task']-=options['task_base']
	if options['time_units'] not in ['seconds','milliseconds']:
		raise ValueError("Unknown supplied units: '{units}'. Available values are seconds and milliseconds".format(units=options['time_units']))
	if options['method'] not in ['full','confidence_only','full_confidence']:
		raise ValueError("Unknown supplied method: '{method}'. Available values are full, confidence_only and full_confidence".format(method=options['method']))
	options['experiment'] = options['experiment'].lower()
	if options['experiment'] not in ['all','luminancia','2afc','auditivo']:
		raise ValueError("Unknown experiment supplied: '{0}'. Available values are 'all', 'luminancia', '2afc' and 'auditivo'".format(options['experiment']))
	else:
		# Switching case to the data_io_cognition case sensitive definition of each experiment
		options['experiment'] = {'all':'all','luminancia':'Luminancia','2afc':'2AFC','auditivo':'Auditivo'}[options['experiment']]
	
	if options['plot_merge'] not in [None,'subjects','sessions','all']:
		raise ValueError("Unknown plot_merge supplied: '{0}'. Available values are None, 'subjects', 'sessions' and 'all'.".format(options['plot_merge']))
	
	return options

if __name__=="__main__":
	options = parse_input()
	if options['debug']:
		logging.basicConfig(level=logging.DEBUG)
	elif options['verbose']:
		logging.basicConfig(level=logging.INFO)
	else:
		logging.basicConfig(level=logging.WARNING)
	save = options['save']
	task = options['task']
	ntasks = options['ntasks']
	if save:
		if task==0 and ntasks==1:
			fname = "fits_cognition_{method}{suffix}".format(method=options['method'],suffix=options['suffix'])
		else:
			fname = "fits_cognition_{method}_{task}_{ntasks}{suffix}".format(method=options['method'],task=task,ntasks=ntasks,suffix=options['suffix'])
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
	
	subjects = io.filter_subjects_list(io.unique_subject_sessions(raw_data_dir),'all_sessions_by_experiment')
	#~ problem_2afc = io.filter_subjects_list(subjects,'experiment_2AFC')[51:54:2]
	#~ problem_aud = io.filter_subjects_list(subjects,'experiment_Auditivo')[51:54:2]
	#~ problem_lum = io.filter_subjects_list(subjects,'experiment_Luminancia')[7:10:2]
	#~ print '2AFC: ',[s.get_key() for s in problem_2afc]
	#~ print 'Aud: ',[s.get_key() for s in problem_aud]
	#~ print 'Lum: ',[s.get_key() for s in problem_lum]
	#~ raise RuntimeError('Testing')
	if options['experiment']!='all':
		subjects = io.filter_subjects_list(subjects,'experiment_'+options['experiment'])
	fitter_plot_handler = None
	for i,s in enumerate(subjects):
		logging.debug('Enumerated {0} subject {1}'.format(i,s))
		if (i-task)%ntasks==0:
			logging.debug('Task will execute for enumerated {0} subject {1}'.format(i,s))
			if options['fit']:
				logging.debug('Flag "fit" was True')
				fitter = Fitter(s,time_units=options['time_units'],method=options['method'],\
					   optimizer=options['optimizer'],decisionPolicyKwArgs=options['dpKwargs'],\
					   suffix=options['suffix'],rt_cutoff=options['rt_cutoff'],\
					   confidence_partition=options['confidence_partition'])
				fit_output = fitter.fit(fixed_parameters=options['fixed_parameters'],\
										start_point=options['start_point'],\
										bounds=options['bounds'],\
										optimizer_kwargs=options['optimizer_kwargs'])
				fitter.save()
				if options['method']=='full':
					logging.debug('Used method "full" to fit the decision parameters. Will now execute "confidence_only" method.')
					parameters = fitter.get_parameters_dict_from_fit_output(fit_output)
					del parameters['high_confidence_threshold']
					del parameters['confidence_map_slope']
					fitter.method = 'confidence_only'
					fit_output = fitter.fit(fixed_parameters=parameters,\
											optimizer_kwargs=options['optimizer_kwargs'])
					fitter.save()
			if options['plot'] or save or options['save_plot_handler'] or options['load_plot_handler']:
				logging.debug('Plot, save save_plot_fitter or load_plot_fitter flags were True.')
				fname = 'fits_cognition/{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}{suffix}.pkl'
				if s._single_session:
					ses = str(s.session)
				else:
					ses = '-'.join([str(ses) for ses in s.session])
				method = options['method']
				try:
					if method=='full':
						try:
							formated_fname = fname.format(experiment=s.experiment,method='confidence_only',name=s.name,session=ses,
								 optimizer=options['optimizer'],suffix=options['suffix'])
							logging.debug('Attempting to load fitter from file "{0}".'.format(formated_fname))
							fitter = load_Fitter_from_file(formated_fname)
						except:
							logging.debug('Failed to load fitter confidence_only output file. Will attempt to load full method output file')
							formated_fname = fname.format(experiment=s.experiment,method='full',name=s.name,session=ses,
								 optimizer=options['optimizer'],suffix=options['suffix'])
							logging.debug('Attempting to load fitter from file "{0}".'.format(formated_fname))
							fitter = load_Fitter_from_file(formated_fname)
					else:
						formated_fname = fname.format(experiment=s.experiment,method=method,name=s.name,session=ses,
									 optimizer=options['optimizer'],suffix=options['suffix'])
						logging.debug('Attempting to load fitter from file "{0}".'.format(formated_fname))
						fitter = load_Fitter_from_file(formated_fname)
				except:
					logging.warning('Failed to load fitter from file. Will continue to next subject.')
					if options['debug']:
						raise
					else:
						continue
				logging.debug('Getting fitter_plot_handler with merge_plot={0}.'.format(options['plot_merge']))
				if not options['load_plot_handler']:
					logging.debug('Will not load Fitter_plot_handler from disk')
					temp = fitter.get_fitter_plot_handler(merge=options['plot_merge'])
					if options['save_plot_handler']:
						logging.debug('Saving Fitter_plot_handler to file={0}.'.format(formated_fname.replace('.pkl','_plot_handler.pkl')))
						temp.save(formated_fname.replace('.pkl','_plot_handler.pkl'))
				else:
					logging.debug('Loading Fitter_plot_handler from file={0}'.format(formated_fname.replace('.pkl','_plot_handler.pkl')))
					f = open(formated_fname.replace('.pkl','_plot_handler.pkl'),'r')
					temp = pickle.load(f)
					f.close()
				logging.debug('Adding Fitter_plot_handlers')
				if fitter_plot_handler is None:
					fitter_plot_handler = temp
				else:
					fitter_plot_handler+= temp
	if options['plot'] or save:
		logging.debug('Plotting results from fitter_plot_handler.')
		fitter_plot_handler.plot(saver=saver,display=options['plot'])
		if save:
			logging.debug('Closing saver.')
			saver.close()
