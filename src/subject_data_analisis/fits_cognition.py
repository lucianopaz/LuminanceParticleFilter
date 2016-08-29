from __future__ import division

import enum, os, sys, math, scipy, pickle, warnings, json
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
	raw_data_dir='/homedtic/lpaz/cognition/raw_data'
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

def load_Fitter_from_file(fname):
	f = open(fname,'r')
	fitter = pickle.load(f)
	f.close()
	return fitter

class Fitter:
	# Initer
	def __init__(self,subjectSession,time_units='seconds',method='full',optimizer='cma',\
				decisionPolicyKwArgs={},suffix='',rt_cutoff=14.):
		self.set_experiment(subjectSession.experiment)
		self.rt_cutoff = float(rt_cutoff)
		self.set_time_units(str(time_units))
		self.set_subjectSession_data(subjectSession)
		self.method = str(method)
		self.optimizer = str(optimizer)
		self.suffix = str(suffix)
		if decisionPolicyArgs or decisionPolicyKwArgs:
			if 'dt' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['dt'] = self._ISI
			if 'model_var' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['model_var'] = self._model_var
			if 'internal_var' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['internal_var'] = self._internal_var
			if 'T' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['T'] = self._T
			if 'iti' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['iti'] = self._iti
			if 'tp' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['tp'] = self._tp
			if 'stim_duration' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['stim_duration'] = self._stim_duration
		self._decisionPolicyKwArgs = decisionPolicyKwArgs
		self.dp = ct.decisionPolicy(**decisionPolicyKwArgs)
	
	# Setters
	def set_experiment(self,experiment):
		self.experiment = str(experiment)
		if self.experiment not in ['Luminancia']:#,'Auditivo','2AFC']:
			raise ValueError('Fitter class does not support the experiment "{0}"'.format(experiment))
	
	def set_subjectSession_data(self,subjectSession):
		self._subjectSession_state = subjectSession.__getstate__()
		dat = subjectSession.load_data()
		
		self.rt = dat[:,1]
		if self.time_units=='milliseconds':
			self.rt*=1e3
		valid_trials = self.rt<self.rt_cutoff
		self.rt = self.rt[valid_trials]
		self.max_RT = np.max(self.rt)
		dat = dat[valid_trials]
		
		if self.experiment=='Luminance':
			self.contrast = dat[:,0]-50.
		else:
			self.contrast = dat[:,0]
		self.performance = dat[:,2]
		self.confidence = dat[:,3]
		self.mu,self.mu_indeces,count = np.unique(self.contrast/self._ISI,return_inverse=True,return_counts=True)
		self.mu_prob = count.astype(np.float64)/np.sum(count.astype(np.float64))
		if self.mu[0]==0:
			mus = np.concatenate((-self.mu[-1:0:-1],self.mu))
			p = np.concatenate((self.mu_prob[-1:0:-1],self.mu_prob))
			p[mus!=0]*=0.5
		else:
			mus = np.concatenate((-self.mu[::-1],self.mu))
			p = np.concatenate((self.mu_prob[::-1],self.mu_prob))*0.5
		
		self._prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	
	def set_time_units(self,time_units='seconds'):
		if time_units not in ('seconds','milliseconds'):
			raise ValueError("Invalid time units '{0}'. Available units are seconds and milliseconds".format(units))
		self.time_units = time_units
		if self.time_units=='milliseconds':
			self.rt_cutoff*= 1e3
		self._tp = 0.
		if self.experiment=='Luminance':
			self._stim_duration = np.inf
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
	
	def __setstate__(self,state):
		self.set_experiment(state['experiment'])
		self.set_time_units = state['time_units']
		self.rt_cutoff = state['rt_cutoff']
		self.set_subjectSession_data(SubjectSession(name=state['subjectSession_state']['name'],
													session=state['subjectSession_state']['session'],
													experiment=self.experiment,
													data_dir=state['subjectSession_state']['data_dir']))
		self.method = state['method']
		self.optimizer = state['optimizer']
		self.suffix = state['suffix']
		self._decisionPolicyKwArgs = state['decisionPolicyKwArgs']
		self.dp = ct.decisionPolicy(**decisionPolicyKwArgs)
		
		if 'fit_arguments' in state.keys():
			self.fixed_parameters = state['fit_arguments']['fixed_parameters']
			self.fitted_parameters = state['fit_arguments']['fitted_parameters']
			self._fit_arguments = state['fit_arguments']
		if 'fit_output' in state.keys():
			self._fit_output = state['fit_output']
	
	# Getters
	def get_parameters_dict(self,x):
		parameters = self.fixed_parameters.copy()
		for index,key in self.fitted_parameters:
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
				 'decisionPolicyKwArgs':self._decisionPolicyKwArgs}
		if hasattr(self,'_fit_arguments'):
			state['fit_arguments'] = self._fit_arguments
		if hasattr(self,'_fit_output'):
			state['fit_output'] = self._fit_output
	
	# Defaults
	def default_fixed_parameters(self):
		if self.experiment=='Luminance':
			return {'internal_var':0.}
		else:
			return {}
	
	def default_start_point(self):
		if self.time_units=='seconds':
			return {'cost':0.2,'dead_time':0.1,'dead_time_sigma':0.5,
					'phase_out_prob':0.1,'internal_var':self._internal_var,
					'high_confidence_threshold':0.5,'confidence_map_slope':1e9}
		else:
			return {'cost':0.0002,'dead_time':100.,'dead_time_sigma':500.,
					'phase_out_prob':0.1,'internal_var':self._internal_var,
					'high_confidence_threshold':0.5,'confidence_map_slope':1e9}
	
	def default_bounds(self):
		if self.time_units=='seconds':
			return {'cost':[0.,10],'dead_time':[0.,0.4],'dead_time_sigma':[0.,3.],
					'phase_out_prob':[0.,1.],'internal_var':[self._internal_var*1e-6,self._internal_var*1e3],
					'high_confidence_threshold':[0.,3.],'confidence_map_slope':[0.,1e12]}
		else:
			return {'cost':[0.,0.01],'dead_time':[0.,400.],'dead_time_sigma':[0.,3000.],
					'phase_out_prob':[0.,1.],'internal_var':[self._internal_var*1e-6,self._internal_var*1e3],
					'high_confidence_threshold':[0.,3.],'confidence_map_slope':[0.,1e12]}
	
	def default_optimizer_kwargs(self):
		if self.optimizer=='cma':
			return {'restarts':1}
		else:
			return {'disp': False, 'maxiter': 1000, 'maxfev': 10000, 'repetitions': 10}
	
	# Main fit method
	def fit(self,fixed_parameters=None,start_point=None,bounds=None,optimizer_kwargs=None,fit_arguments=None):
		if fit_arguments is None:
			if fixed_parameters is None:
				fixed_parameters = self.default_fixed_parameters()
			default_start_point = self.default_start_point()
			if not start_point is None:
				self.default_start_point.update(start_point)
			default_bounds = self.default_bounds()
			if not bounds is None:
				default_bounds.update(bounds)
			start_point,bounds = self.sanitize_x0_bounds(fixed_parameters,start_point,bounds)
			
			default_optimizer_kwargs = self.default_optimizer_kwargs()
			if optimizer_kwargs:
				for key in optimizer_kwargs.keys():
					default_optimizer_kwargs[key] = optimizer_kwargs[key]
			optimizer_kwargs = default_optimizer_kwargs
			self._fit_arguments = {'fixed_parameters':self.fixed_parameters,'fitted_parameters':self.fitted_parameters,\
								   'start_point':start_point,'bounds':bounds,'optimizer_kwargs':optimizer_kwargs}
		else:
			start_point = fit_arguments['start_point']
			bounds = fit_arguments['bounds']
			optimizer_kwargs = fit_arguments['optimizer_kwargs']
			self.fitted_parameters = fit_arguments['fitted_parameters']
			self._fit_arguments = fit_arguments
		
		minimizer = self.init_minimizer(start_point,bounds,optimizer_kwargs)
		if self.experiment=='Luminance':
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
		return self._fit_output
	
	# Savers
	def save_fit_output(self,fit_output):
		state = self.__getstate__()
		if 'fit_output' not in state:
			raise ValueError('The Fitter instance has not performed any fit and still has no _fit_output attribute set')
		session = self._subjectSession_state['session']
		if isinstance(session,int):
			session = str(session)
		else:
			session = '-'.join([str(s) for s in session])
		fname = 'testing/{experiment}_fit_{method}_subject_{name}_session_{session}_{suffix}.pkl'.format(
				experiment=self.experiment,method=self.method,name=self._subjectSession_state['name'],
				session=session,suffix=self.suffix)
		f = open(fname,'w')
		pickle.dump(f,state,pickle.HIGHEST_PROTOCOL)
		f.close()
	
	# Sanitizers
	def sanitize_x0_bounds(self,fixed_parameters,start_point,bounds):
		fittable_parameters = ['cost','dead_time','dead_time_sigma','phase_out_prob','internal_var','high_confidence_threshold','confidence_map_slope']
		confidence_parameters = ['high_confidence_threshold','confidence_map_slope']
		self.fixed_parameters = fixed_parameters
		fitted_parameters = []
		temp_x0 = []
		temp_b = [[],[]]
		for par in fittable_parameters:
			if par not in fixed_parameters.keys():
				fitted_parameters.append(par)
				temp_x0.append(start_point[par])
				temp_b[0].append(bounds[par][0])
				temp_b[1].append(bounds[par][1])
		method_fitted_parameters = {'full_confidence':fitted_parameters,'full':[],'confidence_only':[]}
		method_fixed_parameters = {'full_confidence':fixed_parameters.copy(),'full':fixed_parameters.copy(),'confidence_only':fixed_parameters.copy()}
		method_sp = {'full_confidence':np.array(temp_x0[:]),'full':[],'confidence_only':[]}
		method_b = {'full_confidence':np.array([np.array(temp_b[0]),np.array(temp_b[1])]),'full':[[],[]],'confidence_only':[[],[]]}
		for index,par in fitted_parameters:
			if par not in confidence_parameters:
				method_fitted_parameters['full'].append(par)
				method_fixed_parameters['confidence_only'][par] = temp_x0[index]
				method_sp['full'].append(temp_x0[index])
				method_b['full'][0].append(temp_b[0][index])
				method_b['full'][1].append(temp_b[1][index])
			else:
				method_fitted_parameters['confidence_only'].append(par)
				method_fixed_parameters['full'][par] = temp_x0[index]
				method_sp['confidence_only'].append(temp_x0[index])
				method_b['confidence_only'][0].append(temp_b[0][index])
				method_b['confidence_only'][1].append(temp_b[1][index])
		method_sp['full'] = np.array(method_sp['full'])
		method_b['full'] = np.array(method_sp['full'])
		method_sp['confidence_only'] = np.array(method_b['confidence_only'])
		method_b['confidence_only'] = np.array(method_b['confidence_only'])
		
		sanitized_start_point = method_sp[self.method]
		sanitized_bounds = method_b[self.method]
		self.fitted_parameters = method_fitted_parameters[self.method]
		self.fixed_parameters = method_fixed_parameters[self.method]
		if len(fitted_parameters)==1 and self.optimizer=='cma':
			warnings.warn('CMA is unsuited for optimization of single dimensional parameter spaces. Optimizer was changed to Nelder-Mead')
			self.optimizer = 'Nelder-Mead'
		
		if self.optimizer!='cma':
			sanitized_bounds = [(lb,ub) for lb,ub in zip(sanitized_bounds[0],sanitized_bounds[1])]
		return (sanitized_start_point,sanitized_bounds)
	
	def sanitize_fmin_output(self,output,package='cma'):
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
		if self.optimizer=='cma':
			scaling_factor = bounds[1]-bounds[0]
			options = {'bounds':bounds,'CMA_stds':scaling_factor}
			options.update(optimizer_kwargs)
			options = cma.CMAOptions(options)
			if 'restarts' in optimizer_kwargs.keys():
				restarts = optimizer_kwargs['restarts']
			else:
				restarts = 1
			minimizer = lambda x: self.sanitize_fmin_output(cma.fmin(x,start_point,1./3.,options,restarts=restarts),package='cma')
			minimizer = lambda x: self.sanitize_fmin_output((start_point,None,None,None,None,None,None,None),'cma')
		else:
			repetitions = optimizer_kwargs['repetitions']
			_start_points = [start_point]
			_start_points.append(np.random.rand(repetitions-1,len(start_point))*(bounds[1]-bounds[0])+bounds[0])
			start_point_generator = iter(_start_points)
			minimizer = lambda x: self.sanitize_fmin_output(self.repeat_minimize(x,start_point_generator,bounds=bounds,optimizer_kwargs=optimizer_kwargs),package='scipy')
			minimizer = lambda x: self.sanitize_fmin_output({'xbest':start_point,'funbest':None,'nfev':None,'nit':None,'xmean':None,'xstd':None},'scipy')
		
		return minimizer
	
	def repeat_minimize(self,merit,start_point_generator,bounds,optimizer_kwargs):
		output = {'xs':[],'funs':[],'nfev':0,'nit':0,'xbest':None,'funbest':None,'xmean':None,'xstd':None,'funmean':None,'funstd':None}
		repetitions = 0
		for start_point in start_point_generator:
			print bounds,start_point
			repetitions+=1
			res = scipy.optimize.minimize(merit,start_point, method=optimizer,bounds=bounds,options=optimizer_kwargs)
			print 'round {0} ended. Result: '.format(repetitions),res.fun,res.x
			output['xs'].append(res.x)
			output['funs'].append(res.fun)
			output['nfev']+=res.nfev
			output['nit']+=res.nit
			if output['funbest'] is None or res.fun<output['funbest']:
				output['funbest'] = res.fun
				output['xbest'] = res.x
			print 'Best so far: ',output['funbest'],output['xbest']
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
		return phigh,1.-phigh
	
	# Experiment dependant merits
	# Luminancia experiment
	def lum_full_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		self.dp.set_cost(parameters.cost)
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/self.max_RT
		for index,drift in enumerate(self.mu):
			g1,g2 = add_dead_time(self.dp.rt(drift,bounds=(xub,xlb)),self.dp.dt,parameters['dead_time'],parameters['dead_time_sigma'])
			indeces = self.mu_indeces==index
			for rt,perf in zip(self.rt[indeces],self.performance[indeces]):
				if perf==1:
					nlog_likelihood-=np.log(np.exp(-self.dp.rt_nlog_like(g1,rt))*(1-parameters['phase_out_prob'])+random_rt_likelihood)
				else:
					nlog_likelihood-=np.log(np.exp(-self.dp.rt_nlog_like(g2,rt))*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def lum_confidence_only_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		self.dp.set_cost(parameters.cost)
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.5*parameters['phase_out_prob']/self.max_RT
		phigh,plow = self.high_confidence_mapping(parameters['high_confidence_threshold'])
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			confidence_rt = np.concatenate((phigh*gs,plow*gs))
			g1h,g2h,g1l,g2l = add_dead_time(confidence_rt,self.dp.dt,parameters['dead_time'],parameters['dead_time_sigma'])
			gh = g1h+g2h
			gl = g1l+g2l
			indeces = self.mu_indeces==index
			for rt,conf in zip(self.rt[indeces],self.confidence[indeces]):
				if conf==2:
					nlog_likelihood-=np.log(np.exp(-self.dp.rt_nlog_like(gh,rt))*(1-parameters['phase_out_prob'])+random_rt_likelihood)
				else:
					nlog_likelihood-=np.log(np.exp(-self.dp.rt_nlog_like(gl,rt))*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	def lum_full_confidence_merit(self,x):
		parameters = self.get_parameters_dict(x)
		nlog_likelihood = 0.
		self.dp.set_cost(parameters.cost)
		xub,xlb = self.dp.xbounds()
		random_rt_likelihood = 0.25*parameters['phase_out_prob']/self.max_RT
		phigh,plow = self.high_confidence_mapping(parameters['high_confidence_threshold'])
		
		for index,drift in enumerate(self.mu):
			gs = np.array(self.dp.rt(drift,bounds=(xub,xlb)))
			confidence_rt = np.concatenate((phigh*gs,plow*gs))
			g1h,g2h,g1l,g2l = add_dead_time(confidence_rt,self.dp.dt,parameters['dead_time'],parameters['dead_time_sigma'])
			indeces = self.mu_indeces==index
			for rt,perf,conf in zip(self.rt[indeces],self.performance[indeces],self.confidence[indeces]):
				if perf==1 and conf==1:
					nlog_likelihood-=np.log(np.exp(-self.dp.rt_nlog_like(g1h,rt))*(1-parameters['phase_out_prob'])+random_rt_likelihood)
				elif perf==0 and conf==1:
					nlog_likelihood-=np.log(np.exp(-self.dp.rt_nlog_like(g2h,rt))*(1-parameters['phase_out_prob'])+random_rt_likelihood)
				elif perf==1 and conf==0:
					nlog_likelihood-=np.log(np.exp(-self.dp.rt_nlog_like(g1l,rt))*(1-parameters['phase_out_prob'])+random_rt_likelihood)
				else:
					nlog_likelihood-=np.log(np.exp(-self.dp.rt_nlog_like(g2l,rt))*(1-parameters['phase_out_prob'])+random_rt_likelihood)
		return nlog_likelihood
	
	# 2AFC experiment
	def afc_full_merit(self,parameters):
		return np.nan
	
	def afc_confidence_only_merit(self,parameters):
		return np.nan
	
	def afc_full_confidence_merit(self,parameters):
		return np.nan
	
	# Auditivo experiment
	def aud_full_merit(self,parameters):
		return np.nan
	
	def aud_confidence_only_merit(self,parameters):
		return np.nan
	
	def aud_full_confidence_merit(self,parameters):
		return np.nan
	
	# Theoretical predictions
	def theoretical_rt_distribution(self,fit_output=None,return_confidence=True,include_t0=True):
		parameters = self.get_parameters_dict(fit_output)
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
			if include_t0:
				phased_out_rt[self.dp.t<_max_RT] = 1./(_max_RT)
			else:
				phased_out_rt[self.dp.t[1:]<_max_RT] = 1./(_max_RT)
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
			self.dp.set_cost(cost)
			_max_RT = self.dp.t[np.ceil(max_RT/self.dp.dt)]
			
			phigh,plow = self.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
			phased_out_rt[self.dp.t<_max_RT] = 1./(_max_RT)
			
			for index,drift in enumerate(mu):
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
				rt['full']['high']+= rt[drift]['high']*mu_prob[index]
				rt['full']['low']+= rt[drift]['low']*mu_prob[index]
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
		
		mxlim = np.ceil(max_RT)
		plt.figure(figsize=(11,8))
		ax1 = plt.subplot(121)
		plt.step(xh,hit_rt,label='Subject '+str(subject.id)+' hit rt',where='post',color='b')
		plt.step(xh,-miss_rt,label='Subject '+str(subject.id)+' miss rt',where='post',color='r')
		plt.plot(self.dp.t,sim_rt['full']['all'][0],label='Theoretical hit rt',linewidth=2,color='b')
		plt.plot(self.dp.t,-sim_rt['full']['all'][1],label='Theoretical miss rt',linewidth=2,color='r')
		plt.xlim([0,mxlim])
		if self.time_units=='seconds':
			plt.xlabel('T [s]')
		else:
			plt.xlabel('T [ms]')
		plt.ylabel('Prob density')
		plt.legend()
		plt.subplot(122,sharey=ax1)
		plt.step(xh,high_hit_rt+high_miss_rt,label='Subject '+str(subject.id)+' high',where='post',color='forestgreen')
		plt.step(xh,-(low_hit_rt+low_miss_rt),label='Subject '+str(subject.id)+' low',where='post',color='mediumpurple')
		plt.plot(self.dp.t,np.sum(sim_rt['full']['high'],axis=0),label='Theoretical high',linewidth=2,color='forestgreen')
		plt.plot(self.dp.t,-np.sum(sim_rt['full']['low'],axis=0),label='Theoretical low',linewidth=2,color='mediumpurple')
		plt.xlim([0,mxlim])
		if time_units=='seconds':
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
				'optimizer_kwargs':{}}
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
				expecting_key = True
			elif arg=='--fixed_parameters':
				key = 'fixed_parameters'
				expecting_key = True
				json_encoded_key = True
			elif arg=='--start_point':
				key = 'start_point'
				expecting_key = True
				json_encoded_key = True
			elif arg=='--bounds':
				key = 'bounds'
				expecting_key = True
				json_encoded_key = True
			elif arg=='--dpKwargs':
				key = 'dpKwargs'
				expecting_key = True
				json_encoded_key = True
			elif arg=='--optimizer_kwargs':
				key = 'optimizer_kwargs'
				expecting_key = True
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
	return options

if __name__=="__main__":
	options = parse_input()
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
				fitter.save_fit_output()
				if options['method']=='full':
					parameters = fitter.get_parameters_dict_from_fit_output(fit_output)
					del parameters['high_confidence_threshold']
					del parameters['confidence_map_slope']
					fitter.method = 'confidence_only'
					fit_output = fitter.fit(fixed_parameters=parameters,\
											optimizer_kwargs=options['optimizer_kwargs'])
					fitter.save_fit_output()
			if options['plot'] or save:
				fname = 'testing/{experiment}_fit_{method}_subject_{name}_session_{session}_{suffix}.pkl'
				if s._single_session:
					ses = str(s.session)
				else:
					ses = '-'.join([str(ses) for ses in s.session])
				method = options['method']
				if method=='full':
					method = 'confidence_only'
				fname.format(experiment=s.experiment,method=method,name=self.s.name,session=ses,suffix=options['suffix'])
				fitter = load_Fitter_from_file(formated_fname)
				fitter.plot_fit(saver=savers[s.experiment],display=options['plot'])
	if save:
		for key in savers.keys():
			if not isinstance(savers[key],str):
				savers[key].close()
