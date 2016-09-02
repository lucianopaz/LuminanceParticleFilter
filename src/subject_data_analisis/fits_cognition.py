from __future__ import division

import enum, os, sys, math, scipy, pickle, warnings, json, logging
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
				decisionPolicyKwArgs={},suffix='',rt_cutoff=14.):
		logging.info('Creating Fitter instance')
		self.raw_data_dir = raw_data_dir
		self.set_experiment(subjectSession.experiment)
		self.rt_cutoff = float(rt_cutoff)
		logging.debug('Setted Fitter rt_cutoff = %f',self.rt_cutoff)
		self.set_time_units(str(time_units))
		self.set_subjectSession_data(subjectSession)
		self.method = str(method)
		logging.debug('Setted Fitter method = %s',self.method)
		self.optimizer = str(optimizer)
		logging.debug('Setted Fitter optimizer = %s',self.optimizer)
		self.suffix = str(suffix)
		logging.debug('Setted Fitter suffix = %s',self.suffix)
		self.set_decisionPolicyKwArgs(decisionPolicyKwArgs)
		self.dp = ct.DecisionPolicy(**self.decisionPolicyKwArgs)
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
		if self.time_units=='milliseconds':
			self.rt_cutoff*= 1e3
		
		self._tp = 0.
		logging.debug('Setted Fitter instance _tp = %f',self._tp)
		if self.experiment=='Luminancia':
			logging.debug('Luminancia experiment condition')
			if time_units=='seconds':
				self._ISI = 0.05
				self._T = 10.
				self._iti = 1.5
			else:
				self._ISI = 50.
				self._T = 10000.
				self._iti = 1500.
			self._model_var = 50./self._ISI
			self._internal_var = 0.
			self._fixed_stim_duration = 0.
			logging.debug('Setted _ISI = %(_ISI)f\t_T = %(_T)f\t_iti = %(_iti)f\t_model_var = %(_model_var)f\t_internal_var = %(_internal_var)f\tfixed_stim_duration = %(_fixed_stim_duration)f',
						  {'_ISI':self._ISI,'_T':self._T,'_iti':self._iti,'_model_var':self._model_var,'_internal_var':self._internal_var,'_fixed_stim_duration':self._fixed_stim_duration})
		elif self.experiment=='Auditivo':
			logging.debug('Auditivo experiment condition')
			if time_units=='seconds':
				self._ISI = 0.5
				self._T = 0.3
				self._iti = 2.
			else:
				self._ISI = 500.
				self._T = 300.
				self._iti = 2000.
			self._model_var = 0.
			self._internal_var = 0.
			self._fixed_stim_duration = 0.3
			logging.debug('Setted _ISI = %(_ISI)f\t_T = %(_T)f\t_iti = %(_iti)f\t_model_var = %(_model_var)f\t_internal_var = %(_internal_var)f\tfixed_stim_duration = %(_fixed_stim_duration)f',
						  {'_ISI':self._ISI,'_T':self._T,'_iti':self._iti,'_model_var':self._model_var,'_internal_var':self._internal_var,'_fixed_stim_duration':self._fixed_stim_duration})
		elif self.experiment=='2AFC':
			logging.debug('Auditivo experiment condition')
			if time_units=='seconds':
				self._ISI = 0.3
				self._T = 0.3
				self._iti = 1.5
			else:
				self._ISI = 300.
				self._T = 300.
				self._iti = 1500.
			self._model_var = 0.
			self._internal_var = 0.
			self._fixed_stim_duration = 0.3
			logging.debug('Setted _ISI = %(_ISI)f\t_T = %(_T)f\t_iti = %(_iti)f\t_model_var = %(_model_var)f\t_internal_var = %(_internal_var)f\tfixed_stim_duration = %(_fixed_stim_duration)f',
						  {'_ISI':self._ISI,'_T':self._T,'_iti':self._iti,'_model_var':self._model_var,'_internal_var':self._internal_var,'_fixed_stim_duration':self._fixed_stim_duration})
	
	def set_subjectSession_data(self,subjectSession):
		self._subjectSession_state = subjectSession.__getstate__()
		logging.debug('Setted Fitter _subjectSession_state')
		logging.debug('experiment:%(experiment)s, name:%(name)s, session:%(session)s, data_dir:%(data_dir)s',self._subjectSession_state)
		dat = subjectSession.load_data(override_raw_data_dir={'original':self.raw_data_dir,'replacement':raw_data_dir})
		logging.debug('Loading subjectSession data')
		self.rt = dat[:,1]
		if self.time_units=='milliseconds':
			self.rt*=1e3
		valid_trials = self.rt<self.rt_cutoff
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
		self.mu,self.mu_indeces,count = np.unique(self.contrast/self._ISI,return_inverse=True,return_counts=True)
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
		self.rt_cutoff = state['rt_cutoff']
		self.set_time_units(state['time_units'])
		self.set_subjectSession_data(io.SubjectSession(name=state['subjectSession_state']['name'],
													   session=state['subjectSession_state']['session'],
													   experiment=self.experiment,
													   data_dir=state['subjectSession_state']['data_dir']))
		self.method = state['method']
		self.optimizer = state['optimizer']
		self.suffix = state['suffix']
		self.set_decisionPolicyKwArgs(state['decisionPolicyKwArgs'])
		self.dp = ct.DecisionPolicy(**self.decisionPolicyKwArgs)
		self.raw_data_dir = state['raw_data_dir']
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
	
	def __getstate__(self):
		state = {'experiment':self.experiment,
				 'time_units':self.time_units,
				 'rt_cutoff':self.rt_cutoff,
				 'subjectSession_state':self._subjectSession_state,
				 'method':self.method,
				 'optimizer':self.optimizer,
				 'suffix':self.suffix,
				 'decisionPolicyKwArgs':self.decisionPolicyKwArgs,
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
		dense_conv_val = normpdf(dense_conv_x,parameters['dead_time'],parameters['dead_time_sigma'])
		dense_conv_val[dense_conv_x<parameters['dead_time']] = 0.
		
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
		if self.experiment=='Luminancia':
			return {'internal_var':0.}
		else:
			return {}
	
	def default_start_point(self):
		if self.time_units=='seconds':
			return {'cost':0.2,'dead_time':0.1,'dead_time_sigma':0.5,
					'phase_out_prob':0.1,'internal_var':1250.,
					'high_confidence_threshold':0.5,'confidence_map_slope':1e9}
		else:
			return {'cost':0.0002,'dead_time':100.,'dead_time_sigma':500.,
					'phase_out_prob':0.1,'internal_var':1250.,
					'high_confidence_threshold':0.5,'confidence_map_slope':1e9}
	
	def default_bounds(self):
		if self.time_units=='seconds':
			return {'cost':[0.,10],'dead_time':[0.,0.4],'dead_time_sigma':[0.,3.],
					'phase_out_prob':[0.,1.],'internal_var':[0.,1e5],
					'high_confidence_threshold':[0.,3.],'confidence_map_slope':[0.,1e12]}
		else:
			return {'cost':[0.,0.01],'dead_time':[0.,400.],'dead_time_sigma':[0.,3000.],
					'phase_out_prob':[0.,1.],'internal_var':[0.,1e2],
					'high_confidence_threshold':[0.,3.],'confidence_map_slope':[0.,1e12]}
	
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
	
	def confidence_mapping_pdf_matrix(self,first_passage_pdfs,parameters,mapped_confidences=None,confidence_partition=100,return_unconvoluted_matrix=False):
		indeces = np.arange(0,confidence_partition,dtype=np.float)
		confidence_array = np.linspace(0,1,confidence_partition)
		nT = self.dp.nT
		confidence_matrix = np.zeros((2,confidence_partition,nT))
		if mapped_confidences is None:
			mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		conv_val,conv_x = self.get_dead_time_convolver(parameters)
		_nT = len(conv_x)
		conv_confidence_matrix = np.zeros((2,confidence_partition,nT+_nT))
		for performance,(first_passage_pdf,mapped_confidence) in enumerate(zip(first_passage_pdfs,mapped_confidences)):
			cv_inds = np.interp(mapped_confidence,confidence_array,indeces)
			for index,(cv_ind,floor_cv_ind,ceil_cv_ind,fppdf) in enumerate(zip(cv_inds,np.floor(cv_inds),np.ceil(cv_inds),first_passage_pdf)):
				norm = 0.
				if index==0:
					weight = 1.-np.mod(cv_ind,1)
					confidence_matrix[performance,floor_cv_ind,index] = fppdf*weight
					confidence_matrix[performance,ceil_cv_ind,index] = fppdf*(1.-weight)
					prev_norm = fppdf
				else:
					if np.abs(cv_ind-prior_cv_ind)<=1.5:
						weight = 1.-np.mod(cv_ind,1)
						confidence_matrix[performance,floor_cv_ind,index] = fppdf*weight
						confidence_matrix[performance,ceil_cv_ind,index] = fppdf*(1.-weight)
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
							confidence_matrix[performance,prior_ceil_cv_ind+1:ceil_cv_ind+1,index-1]+= prev_temp_fppdf
							confidence_matrix[performance,prior_ceil_cv_ind+1:ceil_cv_ind+1,index] = curr_temp_fppdf
						else:
							prev_polypars = np.polyfit([prior_cv_ind,floor_cv_ind],[prior_fppdf,0.],1)
							curr_polypars = np.polyfit([prior_cv_ind,floor_cv_ind],[0.,fppdf],1)
							
							prev_temp_fppdf = np.polyval(prev_polypars,np.arange(prior_floor_cv_ind-1,floor_cv_ind-1,-1))
							curr_temp_fppdf = np.polyval(curr_polypars,np.arange(prior_floor_cv_ind-1,floor_cv_ind-1,-1))
							prev_temp_fppdf[prev_temp_fppdf<0.] = 0.
							curr_temp_fppdf[curr_temp_fppdf<0.] = 0.
							
							prev_norm+= np.sum(prev_temp_fppdf)
							norm = np.sum(curr_temp_fppdf)
							confidence_matrix[performance,prior_floor_cv_ind-1:floor_cv_ind-1:-1,index-1]+= prev_temp_fppdf
							confidence_matrix[performance,prior_floor_cv_ind-1:floor_cv_ind-1:-1,index] = curr_temp_fppdf
					if prev_norm>0:
						confidence_matrix[performance,:,index-1]*=prior_fppdf/prev_norm
					if index<len(cv_inds)-1:
						end_index = np.min([_nT+index-1,nT])
						cv_end_index = end_index-index+1
						conv_confidence_matrix[performance,:,index-1:end_index]+= np.reshape(confidence_matrix[performance,:,index-1],(-1,1))*conv_val[:cv_end_index]
					else:
						if norm>0:
							confidence_matrix[performance,:,index]*=fppdf/norm
						end_index = np.min([_nT+index,nT])
						cv_end_index = end_index-index
						conv_confidence_matrix[performance,:,index:end_index]+= np.reshape(confidence_matrix[performance,:,index],(-1,1))*conv_val[:cv_end_index]
					prev_norm = norm
				prior_fppdf = fppdf
				prior_cv_ind = cv_ind
				prior_floor_cv_ind = floor_cv_ind
				prior_ceil_cv_ind = ceil_cv_ind
		if return_unconvoluted_matrix:
			output = (conv_confidence_matrix,confidence_matrix)
		else:
			output = conv_confidence_matrix
		return output
	
	def decision_pdf(self,first_passage_pdfs,parameters):
		conv_val,conv_x = self.get_dead_time_convolver(parameters)
		_nT = len(conv_x)
		decision_pdfs = np.zeros((2,self.dp.nT+_nT))
		for performance,first_passage_pdf in enumerate(first_passage_pdfs):
			for index in range(self.dp.nT-1,-1,-1):
				decision_pdfs[performance,index:index+_nT]+= first_passage_pdf[index]*conv_val
		return decision_pdfs
	
	def mapped_confidence_probability(self,confidence_mapping_pdf_matrix):
		return np.sum(confidence_mapping_pdf_matrix,axis=confidence_mapping_pdf_matrix.ndim)*self.dp.dt
	
	# Experiment dependant merits
	# Luminancia experiment
	def lum_full_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		self.dp.set_cost(parameters['cost'])
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		for index,drift in enumerate(self.mu):
			gs = self.decision_pdf(np.array(self.dp.rt(drift,bounds=(xub,xlb))),parameters)
			t = np.arange(0,gs.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
				#~ temp1 = rt_likelihood(t,gs[int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood
				#~ temp = np.log(temp1)
				#~ if np.isnan(temp) or np.isnan(temp1):
					#~ print parameters,temp,temp1
				nlog_likelihood-= np.log(rt_likelihood(t,gs[int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def lum_confidence_only_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		if self.__fit_internals__ is None:
			self.dp.set_cost(parameters['cost'])
			xub,xlb = self.dp.xbounds()
			must_compute_first_passage_time = True
			self.__fit_internals__ = {'xub':xub,'xlb':xlb,'first_passage_times':{}}
		else:
			must_compute_first_passage_time = False
			first_passage_times = self.__fit_internals__['first_passage_times']
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
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
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.25*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			confidence_likelihood = self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences)
			t = np.arange(0,confidence_likelihood.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,confidence_likelihood[int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
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
				nlog_likelihood-= np.log(rt_likelihood(t,gs[int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
				#~ temp1 = rt_likelihood(t,gs[int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood
				#~ temp = np.log(temp1)
				#~ if np.isnan(temp) or np.isnan(temp1):
					#~ print parameters,temp,temp1
				#~ nlog_likelihood-= temp
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
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
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
		random_rt_likelihood = 0.25*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			confidence_likelihood = self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences)
			t = np.arange(0,confidence_likelihood.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,confidence_likelihood[int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
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
				nlog_likelihood-= np.log(rt_likelihood(t,gs[int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
				#~ temp1 = rt_likelihood(t,gs[int(perf)],rt)*(1-parameters['phase_out_prob'])+random_rt_likelihood
				#~ temp = np.log(temp1)
				#~ if np.isnan(temp) or np.isnan(temp1):
					#~ print parameters,temp,temp1,gs[int(perf)]
				#~ nlog_likelihood-= temp
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
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
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
		random_rt_likelihood = 0.25*parameters['phase_out_prob']/(self.max_RT-self.min_RT)
		mapped_confidences = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			confidence_likelihood = self.confidence_mapping_pdf_matrix(gs,parameters,mapped_confidences=mapped_confidences)
			t = np.arange(0,confidence_likelihood.shape[-1])*self.dp.dt
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				nlog_likelihood-=np.log(rt_confidence_likelihood(t,confidence_likelihood[int(perf)],rt,conf)*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	# Theoretical predictions
	def theoretical_rt_distribution(self,fit_output=None,return_confidence=True,include_t0=True):
		parameters = self.get_parameters_dict_from_fit_output(fit_output)
		rt,dec_gs = self.decision_rt_distribution(parameters,return_gs=True,include_t0=include_t0)
		if return_confidence:
			dec_conf = self.confidence_rt_distribution(parameters,dec_gs,include_t0=include_t0)
			for key in rt.keys():
				rt[key].update(dec_conf[key])
		return rt
	
	def decision_rt_distribution(self,parameters,return_gs=False,include_t0=True):
		if self.experiment=='Luminancia':
			if include_t0:
				rt = {'full':{'all':np.zeros((2,self.dp.t.shape[0]))}}
				gs = {'full':{'all':np.zeros((2,self.dp.t.shape[0]))}}
				phased_out_rt = np.zeros_like(self.dp.t)
			else:
				rt = {'full':{'all':np.zeros((2,self.dp.t.shape[0]-1))}}
				gs = {'full':{'all':np.zeros((2,self.dp.t.shape[0]-1))}}
				phased_out_rt = np.zeros_like(self.dp.t)[1:]
			self.dp.internal_var = parameters['internal_var']
			self.dp.set_cost(parameters['cost'])
			_max_RT = self.dp.t[np.ceil(self.max_RT/self.dp.dt)]
			_min_RT = self.dp.t[np.ceil(self.min_RT/self.dp.dt)]
			if include_t0:
				phased_out_rt[np.logical_and(self.dp.t<=_max_RT,self.dp.t>=_max_RT)] = 1./(_max_RT-_min_RT)
			else:
				phased_out_rt[np.logical_and(self.dp.t[1:]<=_max_RT,self.dp.t[1:]>=_max_RT)] = 1./(_max_RT-_min_RT)
			xub,xlb = self.dp.xbounds()
			for index,drift in enumerate(self.mu):
				g = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
				if not include_t0:
					g = g[:,1:]
				g1,g2 = add_dead_time(g,self.dp.dt,parameters['dead_time'],parameters['dead_time_sigma'])
				g1 = g1*(1-parameters['phase_out_prob'])+0.5*parameters['phase_out_prob']*phased_out_rt
				g2 = g2*(1-parameters['phase_out_prob'])+0.5*parameters['phase_out_prob']*phased_out_rt
				rt[drift] = {}
				rt[drift]['all'] = np.array([g1,g2])
				rt['full']['all']+=rt[drift]['all']*self.mu_prob[index]
				gs[drift] = {}
				gs[drift]['all'] = np.array(g)
				gs['full']['all']+=gs[drift]['all']*self.mu_prob[index]
			output = (rt,)
			if return_gs:
				output+=(gs,)
		else:
			raise ValueError('Decision RT distribution not implemented for experiment "{0}"'.format(self.experiment))
		return output
	
	def confidence_rt_distribution(self,parameters,dec_gs,include_t0=True):
		if self.experiment=='Luminancia':
			if include_t0:
				rt = {'full':{'high':np.zeros((2,self.dp.t.shape[0])),'low':np.zeros((2,self.dp.t.shape[0]))}}
				phased_out_rt = np.zeros_like(self.dp.t)
			else:
				rt = {'full':{'high':np.zeros((2,self.dp.t.shape[0]-1)),'low':np.zeros((2,self.dp.t.shape[0]-1))}}
				phased_out_rt = np.zeros_like(self.dp.t)[1:]
			self.dp.set_cost(parameters['cost'])
			self.dp.set_internal_var(parameters['internal_var'])
			_max_RT = self.dp.t[np.ceil(self.max_RT/self.dp.dt)]
			_min_RT = self.dp.t[np.ceil(self.min_RT/self.dp.dt)]
			if include_t0:
				phased_out_rt[np.logical_and(self.dp.t<=_max_RT,self.dp.t>=_max_RT)] = 1./(_max_RT-_min_RT)
			else:
				phased_out_rt[np.logical_and(self.dp.t[1:]<=_max_RT,self.dp.t[1:]>=_max_RT)] = 1./(_max_RT-_min_RT)
			
			phigh,plow = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
			
			for index,drift in enumerate(self.mu):
				g = dec_gs[drift]['all']
				if include_t0:
					gh=phigh*g
					gl=plow*g
				else:
					gh=phigh[:,1:]*g
					gl=plow[:,1:]*g
				g1h,g2h,g1l,g2l = add_dead_time(np.concatenate((gh,gl)),self.dp.dt,parameters['dead_time'],parameters['dead_time_sigma'])
				g1h = g1h*(1-parameters['phase_out_prob'])+0.25*parameters['phase_out_prob']*phased_out_rt
				g2h = g2h*(1-parameters['phase_out_prob'])+0.25*parameters['phase_out_prob']*phased_out_rt
				g1l = g1l*(1-parameters['phase_out_prob'])+0.25*parameters['phase_out_prob']*phased_out_rt
				g2l = g2l*(1-parameters['phase_out_prob'])+0.25*parameters['phase_out_prob']*phased_out_rt
				rt[drift] = {}
				rt[drift]['high'] = np.array([g1h,g2h])
				rt[drift]['low'] = np.array([g1l,g2l])
				rt['full']['high']+= rt[drift]['high']*self.mu_prob[index]
				rt['full']['low']+= rt[drift]['low']*self.mu_prob[index]
		else:
			raise ValueError('Confidence RT distribution not implemented for experiment "{0}"'.format(self.experiment))
		return rt
	
	# Plotter
	def plot_fit(self,fit_output=None,saver=None,display=True):
		if not can_plot:
			raise ImportError('Could not import matplotlib package and it is imposible to plot fit')
		theo_rt = self.theoretical_rt_distribution(fit_output)
		
		median_confidence = np.median(self.confidence)
		temp,edges = np.histogram(self.rt,100)
		
		high_hit_rt,temp = np.histogram(self.rt[np.logical_and(self.performance==1,self.confidence>=median_confidence)],edges)
		low_hit_rt,temp = np.histogram(self.rt[np.logical_and(self.performance==1,self.confidence<median_confidence)],edges)
		high_miss_rt,temp = np.histogram(self.rt[np.logical_and(self.performance==0,self.confidence>=median_confidence)],edges)
		low_miss_rt,temp = np.histogram(self.rt[np.logical_and(self.performance==0,self.confidence<median_confidence)],edges)
		
		high_hit_rt = high_hit_rt.astype(np.float64)
		low_hit_rt = low_hit_rt.astype(np.float64)
		high_miss_rt = high_miss_rt.astype(np.float64)
		low_miss_rt = low_miss_rt.astype(np.float64)
		hit_rt = high_hit_rt+low_hit_rt
		miss_rt = high_miss_rt+low_miss_rt
		
		xh = np.array([0.5*(x+y) for x,y in zip(edges[1:],edges[:-1])])
		
		normalization = np.sum(hit_rt+miss_rt)*(xh[1]-xh[0])
		hit_rt/=normalization
		miss_rt/=normalization
		
		high_hit_rt/=normalization
		high_miss_rt/=normalization
		low_hit_rt/=normalization
		low_miss_rt/=normalization
		
		mxlim = np.ceil(self.max_RT)
		plt.figure(figsize=(11,8))
		ax1 = plt.subplot(121)
		plt.step(xh,hit_rt,label='Subject '+str(self._subjectSession_state['name'])+' hit rt',where='post',color='b')
		plt.step(xh,-miss_rt,label='Subject '+str(self._subjectSession_state['name'])+' miss rt',where='post',color='r')
		plt.plot(self.dp.t,theo_rt['full']['all'][0],label='Theoretical hit rt',linewidth=2,color='b')
		plt.plot(self.dp.t,-theo_rt['full']['all'][1],label='Theoretical miss rt',linewidth=2,color='r')
		plt.xlim([0,mxlim])
		if self.time_units=='seconds':
			plt.xlabel('T [s]')
		else:
			plt.xlabel('T [ms]')
		plt.ylabel('Prob density')
		plt.legend()
		plt.subplot(122,sharey=ax1)
		plt.step(xh,high_hit_rt+high_miss_rt,label='Subject '+str(self._subjectSession_state['name'])+' high',where='post',color='forestgreen')
		plt.step(xh,-(low_hit_rt+low_miss_rt),label='Subject '+str(self._subjectSession_state['name'])+' low',where='post',color='mediumpurple')
		plt.plot(self.dp.t,np.sum(theo_rt['full']['high'],axis=0),label='Theoretical high',linewidth=2,color='forestgreen')
		plt.plot(self.dp.t,-np.sum(theo_rt['full']['low'],axis=0),label='Theoretical low',linewidth=2,color='mediumpurple')
		plt.xlim([0,mxlim])
		if self.time_units=='seconds':
			plt.xlabel('T [s]')
		else:
			plt.xlabel('T [ms]')
		plt.legend()
		
		
		if saver:
			if isinstance(saver,str):
				plt.savefig(saver,bbox_inches='tight')
			else:
				saver.savefig()
		if display:
			plt.show(True)

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
                loading subject data. [Default 14 seconds]
 '--merge': Can be None, 'all', 'all_sessions' or 'all_subjects'. This parameter
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
				'plot':False,'fit':True,'time_units':'seconds','suffix':'','rt_cutoff':14.,
				'merge':None,'fixed_parameters':{},'dpKwargs':{},'start_point':{},'bounds':{},
				'optimizer_kwargs':{},'experiment':'all','debug':False}
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
			elif arg=='--plot':
				options['plot'] = True
			elif arg=='-g' or arg=='--debug':
				options['debug'] = True
			elif arg=='--fit':
				options['fit'] = True
			elif arg=='--no-fit':
				options['fit'] = False
			elif arg=='-u' or arg=='--units':
				key = 'time_units'
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
			elif arg=='-h' or arg=='--help':
				print script_help
				sys.exit()
			else:
				raise RuntimeError("Unknown option: {opt} encountered in position {pos}. Refer to the help to see the list of options".format(opt=arg,pos=i+1))
		else:
			expecting_key = True
			if key in ['task','ntasks','task_base']:
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
	return options

if __name__=="__main__":
	options = parse_input()
	if options['debug']:
		logging.basicConfig(level=logging.DEBUG)
	else:
		logging.basicConfig(level=logging.INFO)
	save = options['save']
	task = options['task']
	ntasks = options['ntasks']
	if save:
		if task==0 and ntasks==1:
			fname = "fit_{method}{suffix}".format(method=method,suffix=options['suffix'])
		else:
			fname = "fit_{method}_{task}_{ntasks}{suffix}".format(method=method,task=task,ntasks=ntasks,suffix=options['suffix'])
		if os.path.isdir("../../figs"):
			fname = "../../figs/"+fname
		if loc==Location.cluster:
			fname+='.png'
			savers = {'Luminancia':'Luminancia_'+fname,'Auditivo':'Auditivo_'+fname,'2AFC':'2AFC_'+fname}
		else:
			fname+='.pdf'
			savers = {'Luminancia':PdfPages('Luminancia_'+fname),'Auditivo':PdfPages('Auditivo_'+fname),'2AFC':PdfPages('2AFC_'+fname)}
	else:
		savers = {'Luminancia':None,'Auditivo':None,'2AFC':None}
	
	subjects = io.filter_subjects_list(io.unique_subject_sessions(raw_data_dir),'all_sessions_by_experiment')
	if options['experiment']!='all':
		subjects = io.filter_subjects_list(subjects,'experiment_'+options['experiment'])
	for i,s in enumerate(subjects):
		if (i-task)%ntasks==0:
			if options['fit']:
				fitter = Fitter(s,time_units=options['time_units'],method=options['method'],\
					   optimizer=options['optimizer'],decisionPolicyKwArgs=options['dpKwargs'],\
					   suffix=options['suffix'],rt_cutoff=options['rt_cutoff'])
				fit_output = fitter.fit(fixed_parameters=options['fixed_parameters'],\
										start_point=options['start_point'],\
										bounds=options['bounds'],\
										optimizer_kwargs=options['optimizer_kwargs'])
				fitter.save()
				if options['method']=='full':
					parameters = fitter.get_parameters_dict_from_fit_output(fit_output)
					del parameters['high_confidence_threshold']
					del parameters['confidence_map_slope']
					fitter.method = 'confidence_only'
					fit_output = fitter.fit(fixed_parameters=parameters,\
											optimizer_kwargs=options['optimizer_kwargs'])
					fitter.save()
			if options['plot'] or save:
				fname = 'fits_cognition/{experiment}_fit_{method}_subject_{name}_session_{session}_{optimizer}{suffix}.pkl'
				if s._single_session:
					ses = str(s.session)
				else:
					ses = '-'.join([str(ses) for ses in s.session])
				method = options['method']
				if method=='full':
					method = 'confidence_only'
				formated_fname = fname.format(experiment=s.experiment,method=method,name=s.name,session=ses,
							 optimizer=options['optimizer'],suffix=options['suffix'])
				fitter = load_Fitter_from_file(formated_fname)
				fitter.plot_fit(saver=savers[s.experiment],display=options['plot'])
	if save:
		for key in savers.keys():
			if not isinstance(savers[key],str):
				savers[key].close()
