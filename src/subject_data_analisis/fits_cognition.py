from __future__ import division

import enum, os, sys, math, scipy, pickle
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

class Fitter:
	# Initer
	def __init__(self,subjectSession,time_units='seconds',method='full',optimizer='cma',decisionPolicyArgs=(),\
				decisionPolicyKwArgs={},suffix='',rt_cutoff=14.):
		self.experiment = subjectSession.experiment
		self.rt_cutoff = rt_cutoff
		self.set_time_units(time_units)
		self.set_subjectSession_data(subjectSession)
		self.method = method
		self.optimizer = optimizer
		self.suffix = suffix
		if decisionPolicyArgs or decisionPolicyKwArgs:
			if 'dt' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['dt'] = self._ISI
			if 'model_var' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['model_var'] = self._model_var
			if 'T' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['T'] = self._T
			if 'iti' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['iti'] = self._iti
			if 'tp' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['tp'] = self._tp
			if 'stim_duration' not in decisionPolicyKwArgs:
				decisionPolicyKwArgs['stim_duration'] = self._stim_duration
		self._dp_arguments = {'decisionPolicyArgs':decisionPolicyArgs,
							  'decisionPolicyKwArgs':decisionPolicyKwArgs}
		self.dp = ct.decisionPolicy(*decisionPolicyArgs,**decisionPolicyKwArgs)
	
	# Setters
	def set_subjectSession_data(self,subjectSession):
		self._subjectSession_state = subjectSession.__getstate__()
		dat = subjectSession.load_data()
		
		self.rt = dat[:,1]
		if self.time_units=='milliseconds':
			self.rt*=1e3
		valid_trials = rt<self.rt_cutoff
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
		if mu[0]==0:
			mus = np.concatenate((-mu[-1:0:-1],mu))
			p = np.concatenate((self.mu_prob[-1:0:-1],self.mu_prob))
			p[mus!=0]*=0.5
		else:
			mus = np.concatenate((-mu[::-1],mu))
			p = np.concatenate((self.mu_prob[::-1],self.mu_prob))*0.5
		
		self._prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	
	def set_time_units(time_units='seconds'):
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
	
	# Getters
	def get_parameters_dict(self,x):
		parameters = self.fixed_parameters.copy()
		for index,key in self.fitted_parameters:
			parameters[key] = x[index]
		return parameters
	
	# Defaults
	def default_fixed_parameters(self):
		return {}
	
	def default_start_point(self):
		if self.time_units=='seconds':
			return = [0.2,0.1,0.5,0.1,self._internal_var,0.5,np.inf]
		else:
			return = [0.0002,100,500,0.1,self._internal_var,0.5,np.inf]
	
	def default_bounds(self):
		if self.time_units=='seconds':
			bounds = np.array([np.array([0.,0.,0.,0.,0.]),np.array([10.,0.4,3.,1.,self._internal_var*50,1.])])
		else:
			bounds = np.array([np.array([0.,0.,0.,0.,0.]),np.array([0.01,400,3000,self._internal_var*1e-6,1.])])
	
	def default_optimizer_kwargs(self):
		if self.optimizer=='cma':
			return {'restarts':1}
		else:
			return {'disp': False, 'maxiter': 1000, 'maxfev': 10000, 'repetitions': 10}
	
	# Main fit method
	def fit(self,fixed_parameters=None,start_point=None,bounds=None,optimizer_kwargs=None):
		if fixed_parameters is None:
			fixed_parameters = self.default_fixed_parameters()
		if start_point is None:
			start_point = self.default_start_point()
		if bounds is None:
			bounds = self.default_bounds()
		start_point,bounds = self.sanitize_x0_bounds(fixed_parameters,start_point,bounds)
		start_point = start_point[self.method]
		bounds = bounds[self.method]
		
		default_optimizer_kwargs = self.default_optimizer_kwargs()
		if optimizer_kwargs:
			for key in optimizer_kwargs.keys():
				default_optimizer_kwargs[key] = optimizer_kwargs[key]
		optimizer_kwargs = default_optimizer_kwargs
		self._fit_arguments = {'fitted_args':fitted_args,'start_point':start_point,\
							   'bounds':bounds,'optimizer_kwargs':optimizer_kwargs}
		
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
	def save_fit_output(self):
		detailed_output = {'experiment':self.experiment,\
						   'time_units':self.time_units,\
						   'rt_cutoff':self.rt_cutoff,\
						   'subjectSession_state':self:_subjectSession_state,\
						   'method':self.method,\
						   'optimizer':self.optimizer,\
						   'decisionPolicyArgs':self._dp_arguments['decisionPolicyArgs'],\
						   'decisionPolicyKwArgs':self._dp_arguments['decisionPolicyKwArgs'],\
						   'fit_arguments':self._fit_arguments,\
						   'fit_output':self._fit_output}
		session = self._subjectSession_state['session']
		if isinstance(session,int):
			session = str(session)
		else:
			session = '-'.join([str(s) for s in session])
		fname = 'fits/{experiment}_fit_{method}_subject_{name}_session_{session}_{suffix}.pkl'.format(
				experiment=self.experiment,method=self.method,name=self._subjectSession_state['name'],
				session=session,suffix=self.suffix)
		f = open(fname,'w')
		pickle.dump(f,detailed_output,pickle.HIGHEST_PROTOCOL)
		f.close()
	
	# Sanitizers
	def sanitize_x0_bounds(self,fixed_parameters,start_point,bounds):
		fittable_parameters = ['cost','dead_time','dead_time_sigma','phase_out_prob','model_var','high_confidence_threshold','confidence_map_slope']
		confidence_parameters = ['high_confidence_threshold','confidence_map_slope']
		if len(start_point)!=len(bounds[0]) or len(bounds[0])!=len(bounds[1]):
			raise ValueError('Supplied fixed_parameters must have the same number of columns as bounds')
		if (len(start_point)+len(fixed_parameters.keys()))!=len(fittable_parameters):
			raise ValueError('Inconsistent number of fixed_parameters and start_points supplied')
		self.fixed_parameters = fixed_parameters
		sane_index = 0
		self.fitted_parameters = []
		temp_x0 = []
		temp_b = [[],[]]
		for par in fittable_parameters:
			if par not in fixed_parameters.keys():
				self.fitted_parameters.append(par)
				temp_x0.append(start_point[sane_index])
				temp_b[0].append(bounds[0])
				temp_b[1].append(bounds[1])
				sane_index+=1
		full_confidence_method_sp = np.array(temp_x0[:])
		full_confidence_method_b = np.array([np.array(temp_b[0]),np.array(temp_b[1])])
		full_method_sp = []
		full_method_b = [[],[]]
		confidence_only_method_sp = []
		confidence_only_method_b = [[],[]]
		for index,par in self.fitted_parameters:
			if par not in confidence_parameters:
				full_method_sp.append(temp_x0[index])
				full_method_b[0].append(temp_b[0][index])
				full_method_b[1].append(temp_b[1][index])
			else:
				confidence_only_method_sp.append(temp_x0[index])
				confidence_only_method_b[0].append(temp_b[0][index])
				confidence_only_method_b[1].append(temp_b[1][index])
		full_method_sp = np.array(full_method_sp)
		full_method_b = np.array(full_method_b)
		confidence_only_method_sp = np.array(confidence_only_method_sp)
		confidence_only_method_b = np.array(confidence_only_method_b)
		if self.optimizer!='cma':
			full_method_b = [(lb,ub) for lb,ub in zip(full_method_b[0],full_method_b[1])]
			confidence_only_method_b = [(lb,ub) for lb,ub in zip(confidence_only_method_b[0],confidence_only_method_b[1])]
			full_confidence_method_b = [(lb,ub) for lb,ub in zip(full_confidence_method_b[0],full_confidence_method_b[1])]
		
		sanitized_start_point = {'full':full_method_sp,'full_confidence':full_confidence_method_sp,'confidence_only':confidence_only_method_sp}
		sanitized_bounds = {'full':full_method_b,'full_confidence':full_confidence_method_b,'confidence_only':confidence_only_method_b}
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
			return (fitted_x,output['funbest'],output.nfev,output.nfev,output.nit,output['xmean'],output['xstd'])
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
			return lambda x: self.sanitize_fmin_output(cma.fim(x,start_point,1./3.,options,restarts=restarts),package='cma')
		else:
			repetitions = optimizer_kwargs['repetitions']
			_start_points = [start_point]
			_start_points.append(np.random.rand(repetitions-1,len(start_point))*(bounds[1]-bounds[0])+bounds[0])
			start_point_generator = iter(_start_points)
			return lambda x: self.sanitize_fmin_output(self.repeat_minimize(x,start_point_generator,bounds=bounds,optimizer_kwargs))
	
	def repeat_minimize(self,merit,start_point_generator,bounds=bounds,optimizer_kwargs):
		output = {'xs':[],'funs':[],'nfev':0,'nit':0,'xbest':None,'funbest':None,'xmean':None,'xstd':None,'funmean':None,'funstd':None}
		repetitions = 0
		for start_point in start_point_generator:
			print bounds,start_point
			repetitions+=1
			res = scipy.optimize.minimize(merit,start_point, method=optimizer,bounds=bounds,options=options)
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
	
	
#################################################################################################
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

# Global functions
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

#~ def theoretical_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,confidence_params=None,include_t0=True):
	#~ rt,dec_gs = decision_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,return_gs=True,include_t0=include_t0)
	#~ if confidence_params:
		#~ dec_conf = confidence_rt_distribution(dec_gs,cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,confidence_params,include_t0=include_t0)
		#~ for key in rt.keys():
			#~ rt[key].update(dec_conf[key])
	#~ return rt
#~ 
#~ def decision_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,return_gs=False,include_t0=True):
	#~ if include_t0:
		#~ rt = {'full':{'all':np.zeros((2,m.t.shape[0]))}}
		#~ gs = {'full':{'all':np.zeros((2,m.t.shape[0]))}}
		#~ phased_out_rt = np.zeros_like(m.t)
	#~ else:
		#~ rt = {'full':{'all':np.zeros((2,m.t.shape[0]-1))}}
		#~ gs = {'full':{'all':np.zeros((2,m.t.shape[0]-1))}}
		#~ phased_out_rt = np.zeros_like(m.t)[1:]
	#~ m.set_cost(cost)
	#~ _max_RT = m.t[np.ceil(max_RT/m.dt)]
	#~ _dead_time = m.t[np.floor(dead_time/m.dt)]
	#~ if include_t0:
		#~ phased_out_rt[m.t<_max_RT] = 1./(_max_RT)
	#~ else:
		#~ phased_out_rt[m.t[1:]<_max_RT] = 1./(_max_RT)
	#~ xub,xlb = m.xbounds()
	#~ for index,drift in enumerate(mu):
		#~ g = np.array(m.rt(drift,bounds=(xub,xlb)))
		#~ if not include_t0:
			#~ g = g[:,1:]
		#~ g1,g2 = add_dead_time(g,m.dt,dead_time,dead_time_sigma)
		#~ g1 = g1*(1-phase_out_prob)+0.5*phase_out_prob*phased_out_rt
		#~ g2 = g2*(1-phase_out_prob)+0.5*phase_out_prob*phased_out_rt
		#~ rt[drift] = {}
		#~ rt[drift]['all'] = np.array([g1,g2])
		#~ rt['full']['all']+=rt[drift]['all']*mu_prob[index]
		#~ gs[drift] = {}
		#~ gs[drift]['all'] = np.array(g)
		#~ gs['full']['all']+=gs[drift]['all']*mu_prob[index]
	#~ output = (rt,)
	#~ if return_gs:
		#~ output+=(gs,)
	#~ return output
#~ 
#~ def confidence_rt_distribution(dec_gs,cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,confidence_params,include_t0=True):
	#~ if include_t0:
		#~ rt = {'full':{'high':np.zeros((2,m.t.shape[0])),'low':np.zeros((2,m.t.shape[0]))}}
		#~ phased_out_rt = np.zeros_like(m.t)
	#~ else:
		#~ rt = {'full':{'high':np.zeros((2,m.t.shape[0]-1)),'low':np.zeros((2,m.t.shape[0]-1))}}
		#~ phased_out_rt = np.zeros_like(m.t)[1:]
	#~ m.set_cost(cost)
	#~ _max_RT = m.t[np.ceil(max_RT/m.dt)]
	#~ _dead_time = m.t[np.floor(dead_time/m.dt)]
	#~ 
	#~ if time_units=='seconds' and m.dt>1e-3:
		#~ _dt = 1e-3
	#~ elif time_units=='milliseconds' and m.dt>1.:
		#~ _dt = 1.
	#~ else:
		#~ _dt = None
	#~ 
	#~ if _dt:
		#~ _nT = int(m.T/_dt)+1
		#~ _t = np.arange(0.,_nT,dtype=np.float64)*_dt
		#~ log_odds = m.log_odds()
		#~ log_odds = np.array([np.interp(_t,m.t,log_odds[0]),np.interp(_t,m.t,log_odds[1])])
	#~ else:
		#~ _nT = m.nT
		#~ log_odds = m.log_odds()
	#~ 
	#~ phased_out_rt[m.t<_max_RT] = 1./(_max_RT)
	#~ phigh = np.ones((2,_nT))
	#~ phigh[(log_odds.T/np.max(log_odds,axis=1)).T<confidence_params[0]] = 0.
	#~ if _dt:
		#~ ratio = int(np.ceil(_nT/m.nT))
		#~ tail = _nT%m.nT
		#~ if tail!=0:
			#~ padded_phigh = np.concatenate((phigh,np.nan*np.ones((2,m.nT-tail),dtype=np.float)),axis=1)
		#~ else:
			#~ padded_phigh = phigh
		#~ padded_phigh = np.reshape(padded_phigh,(2,-1,ratio))
		#~ phigh = np.nanmean(padded_phigh,axis=2)
	#~ plow = 1.-phigh
	#~ 
	#~ for index,drift in enumerate(mu):
		#~ g = dec_gs[drift]['all']
		#~ if include_t0:
			#~ gh=phigh*g
			#~ gl=plow*g
		#~ else:
			#~ gh=phigh[:,1:]*g
			#~ gl=plow[:,1:]*g
		#~ g1h,g2h,g1l,g2l = add_dead_time(np.concatenate((gh,gl)),m.dt,dead_time,dead_time_sigma)
		#~ g1h = g1h*(1-phase_out_prob)+0.25*phase_out_prob*phased_out_rt
		#~ g2h = g2h*(1-phase_out_prob)+0.25*phase_out_prob*phased_out_rt
		#~ g1l = g1l*(1-phase_out_prob)+0.25*phase_out_prob*phased_out_rt
		#~ g2l = g2l*(1-phase_out_prob)+0.25*phase_out_prob*phased_out_rt
		#~ rt[drift] = {}
		#~ rt[drift]['high'] = np.array([g1h,g2h])
		#~ rt[drift]['low'] = np.array([g1l,g2l])
		#~ rt['full']['high']+= rt[drift]['high']*mu_prob[index]
		#~ rt['full']['low']+= rt[drift]['low']*mu_prob[index]
	#~ return rt
#~ 
#~ def plot_fit(subject,method='full',save=None,display=True,suffix=''):
	#~ if not can_plot:
		#~ warnings.warn('Unable to plot. Matplotlib package could not be loaded')
		#~ return None
	#~ if method!='confidence_only':
		#~ f = open('fits/inference_fit_'+method+'_subject_'+str(subject.id)+suffix+'.pkl','r')
		#~ out = pickle.load(f)
		#~ f.close()
		#~ if isinstance(out,dict):
			#~ fit_output = out['fit_output']
			#~ options = out['options']
			#~ cost = fit_output[0]['cost']
			#~ dead_time = fit_output[0]['dead_time']
			#~ dead_time_sigma = fit_output[0]['dead_time_sigma']
			#~ phase_out_prob = fit_output[0]['phase_out_prob']
			#~ try:
				#~ high_conf_thresh = fit_output[0]['high_confidence_threshold']
			#~ except KeyError:
				#~ try:
					#~ f = open("fits/inference_fit_confidence_only_subject_"+str(subject.id)+suffix+".pkl",'r')
					#~ out2 = pickle.load(f)
					#~ f.close()
					#~ if isinstance(out2,dict):
						#~ out2 = out2['fit_output']
					#~ high_conf_thresh = out2[0]['high_confidence_threshold']
				#~ except (IOError,EOFError):
					#~ hand_picked_thresh = [0.77,0.6,0.79,0.61,0.62,0.90,0.81]
					#~ high_conf_thresh = hand_picked_thresh[subject.id]
				#~ except Exception as err:
					#~ high_conf_thresh = out2[0][0]
		#~ else:
			#~ fit_output = out
			#~ options = {'time_units':'seconds','T':10.,'iti':1.,'tp':0.,'dt':ISI,'reward':1,'penalty':0,'n':101}
			#~ try:
				#~ cost = fit_output[0]['cost']
				#~ dead_time = fit_output[0]['dead_time']
				#~ dead_time_sigma = fit_output[0]['dead_time_sigma']
				#~ phase_out_prob = fit_output[0]['phase_out_prob']
				#~ try:
					#~ high_conf_thresh = fit_output[0]['high_confidence_threshold']
				#~ except KeyError:
					#~ try:
						#~ f = open("fits/inference_fit_confidence_only_subject_"+str(subject.id)+suffix+".pkl",'r')
						#~ out2 = pickle.load(f)
						#~ f.close()
						#~ high_conf_thresh = out2[0]['high_confidence_threshold']
					#~ except (IOError,EOFError):
						#~ hand_picked_thresh = [0.77,0.6,0.79,0.61,0.62,0.90,0.81]
						#~ high_conf_thresh = hand_picked_thresh[subject.id]
					#~ except Exception as err:
						#~ high_conf_thresh = out2[0][0]
			#~ except IndexError,TypeError:
				#~ cost = fit_output[0][0]
				#~ dead_time = fit_output[0][1]
				#~ dead_time_sigma = fit_output[0][2]
				#~ phase_out_prob = fit_output[0][3]
				#~ if 'confidence' in method:
					#~ high_conf_thresh = fit_output[0][4]
	#~ else:
		#~ f = open('fits/inference_fit_full_subject_'+str(subject.id)+suffix+'.pkl','r')
		#~ out = pickle.load(f)
		#~ f.close()
		#~ if isinstance(out,dict):
			#~ fit_output = out['fit_output']
			#~ options = out['options']
			#~ cost = fit_output[0]['cost']
			#~ dead_time = fit_output[0]['dead_time']
			#~ dead_time_sigma = fit_output[0]['dead_time_sigma']
			#~ phase_out_prob = fit_output[0]['phase_out_prob']
		#~ else:
			#~ fit_output = out
			#~ cost = fit_output[0]['cost']
			#~ dead_time = fit_output[0]['dead_time']
			#~ dead_time_sigma = fit_output[0]['dead_time_sigma']
			#~ phase_out_prob = fit_output[0]['phase_out_prob']
		#~ f = open('fits/inference_fit_confidence_only_subject_'+str(subject.id)+suffix+'.pkl','r')
		#~ out = pickle.load(f)
		#~ f.close()
		#~ if isinstance(out,dict):
			#~ fit_output = out['fit_output']
			#~ options = out['options']
			#~ high_conf_thresh = fit_output[0]['high_confidence_threshold']
		#~ else:
			#~ fit_output = out
			#~ options = {'time_units':'seconds','T':10.,'iti':1.,'tp':0.,'dt':ISI,'reward':1,'penalty':0,'n':101}
			#~ high_conf_thresh = out[0]['high_confidence_threshold']
	#~ time_units = options['time_units']
	#~ set_time_units(options['time_units'])
	#~ 
	#~ dat,t,d = subject.load_data()
	#~ if time_units=='seconds':
		#~ rt = dat[:,1]*1e-3
	#~ else:
		#~ rt = dat[:,1]
	#~ max_RT = np.max(rt)
	#~ perf = dat[:,2]
	#~ conf = dat[:,3]
	#~ temp,edges = np.histogram(rt,100)
	#~ 
	#~ high_hit_rt,temp = np.histogram(rt[np.logical_and(perf==1,conf==2)],edges)
	#~ low_hit_rt,temp = np.histogram(rt[np.logical_and(perf==1,conf==1)],edges)
	#~ high_miss_rt,temp = np.histogram(rt[np.logical_and(perf==0,conf==2)],edges)
	#~ low_miss_rt,temp = np.histogram(rt[np.logical_and(perf==0,conf==1)],edges)
	#~ 
	#~ high_hit_rt = high_hit_rt.astype(np.float64)
	#~ low_hit_rt = low_hit_rt.astype(np.float64)
	#~ high_miss_rt = high_miss_rt.astype(np.float64)
	#~ low_miss_rt = low_miss_rt.astype(np.float64)
	#~ 
	#~ hit_rt = high_hit_rt+low_hit_rt
	#~ miss_rt = high_miss_rt+low_miss_rt
	#~ 
	#~ xh = np.array([0.5*(x+y) for x,y in zip(edges[1:],edges[:-1])])
	#~ 
	#~ normalization = np.sum(hit_rt+miss_rt)*(xh[1]-xh[0])
	#~ hit_rt/=normalization
	#~ miss_rt/=normalization
	#~ 
	#~ high_hit_rt/=normalization
	#~ high_miss_rt/=normalization
	#~ low_hit_rt/=normalization
	#~ low_miss_rt/=normalization
	#~ 
	#~ mu,mu_indeces,count = np.unique((dat[:,0]-distractor)/ISI,return_inverse=True,return_counts=True)
	#~ mu_prob = count.astype(np.float64)
	#~ mu_prob/=np.sum(mu_prob)
	#~ mus = np.concatenate((-mu[::-1],mu))
	#~ counts = np.concatenate((count[::-1].astype(np.float64),count.astype(np.float64)))*0.5
	#~ p = counts/np.sum(counts)
	#~ prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	#~ 
	#~ T = options['T']
	#~ dt = options['dt']
	#~ iti = options['iti']
	#~ tp = options['tp']
	#~ reward = options['reward']
	#~ penalty = options['penalty']
	#~ n = options['n']
	#~ if time_units=='seconds':
		#~ if T is None:
			#~ T = 10.
		#~ if iti is None:
			#~ iti = 1.
		#~ if tp is None:
			#~ tp = 0.
	#~ else:
		#~ if T is None:
			#~ T = 10000.
		#~ if iti is None:
			#~ iti = 1000.
		#~ if tp is None:
			#~ tp = 0.
	#~ if dt is None:
		#~ dt = ISI
	#~ 
	#~ m = ct.DecisionPolicy(model_var=model_var,prior_mu_var=prior_mu_var,n=n,T=T,dt=dt,reward=reward,penalty=penalty,iti=iti,tp=tp,store_p=False)
	#~ 
	#~ confidence_params = [high_conf_thresh]
	#~ sim_rt = theoretical_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,confidence_params)
	#~ 
	#~ mxlim = np.ceil(max_RT)
	#~ mt.rc('axes', color_cycle=['b','r'])
	#~ plt.figure(figsize=(11,8))
	#~ ax1 = plt.subplot(121)
	#~ plt.step(xh,hit_rt,label='Subject '+str(subject.id)+' hit rt',where='post',color='b')
	#~ plt.step(xh,-miss_rt,label='Subject '+str(subject.id)+' miss rt',where='post',color='r')
	#~ plt.plot(m.t,sim_rt['full']['all'][0],label='Theoretical hit rt',linewidth=2,color='b')
	#~ plt.plot(m.t,-sim_rt['full']['all'][1],label='Theoretical miss rt',linewidth=2,color='r')
	#~ plt.xlim([0,mxlim])
	#~ if time_units=='seconds':
		#~ plt.xlabel('T [s]')
	#~ else:
		#~ plt.xlabel('T [ms]')
	#~ plt.ylabel('Prob density')
	#~ plt.legend()
	#~ plt.subplot(122,sharey=ax1)
	#~ plt.step(xh,high_hit_rt+high_miss_rt,label='Subject '+str(subject.id)+' high',where='post',color='forestgreen')
	#~ plt.step(xh,-(low_hit_rt+low_miss_rt),label='Subject '+str(subject.id)+' low',where='post',color='mediumpurple')
	#~ plt.plot(m.t,np.sum(sim_rt['full']['high'],axis=0),label='Theoretical high',linewidth=2,color='forestgreen')
	#~ plt.plot(m.t,-np.sum(sim_rt['full']['low'],axis=0),label='Theoretical low',linewidth=2,color='mediumpurple')
	#~ plt.xlim([0,mxlim])
	#~ if time_units=='seconds':
		#~ plt.xlabel('T [s]')
	#~ else:
		#~ plt.xlabel('T [ms]')
	#~ plt.legend()
	#~ 
	#~ 
	#~ 
	#~ if save:
		#~ if isinstance(save,str):
			#~ plt.savefig(save,bbox_inches='tight')
		#~ else:
			#~ save.savefig()
	#~ if display:
		#~ plt.show(True)

def parse_input():
	script_help = """ moving_bounds_fits.py help
 Sintax:
 moving_bounds_fits.py [option flag] [option value]
 
 moving_bounds_fits.py -h [or --help] displays help
 
 Optional arguments are:
 '-t' or '--task': Integer that identifies the task number when running multiple tasks in parallel. Is one based, thus the first task is task 1 [default 1]
 '-nt' or '--ntasks': Integer that identifies the number tasks working in parallel [default 1]
 '-m' or '--method': String that identifies the fit method. Available values are full, confidence_only and full_confidence. [default full]
 '-o' or '--optimizer': String that identifies the optimizer used for fitting. Available values are 'cma' and all the scipy.optimize.minimize methods. [default cma]
 '-s' or '--save': This flag takes no values. If present it saves the figure.
 '--plot': This flag takes no values. If present it displays the plotted figure and freezes execution until the figure is closed.
 '--fit': This flag takes no values. If present it performs the fit for the selected method. By default, this flag is always set.
 '--no-fit': This flag takes no values. If present no fit is performed for the selected method. This flag should be used when it is only necesary to plot the results.
 '-u' or '--units': String that identifies the time units that will be used. Available values are seconds and milliseconds. [default seconds]
 '-n': Integer that specifies the belief space discretization for the DecisionPolicy instance. Must be an uneven number, if an even number is supplied it will be recast to closest, larger uneven integer (e.g. if n=100 then it will be casted to n=101) [Default 101]
 '-T': Float that specifies the maximum time for the DecisionPolicy instance. [Default 10 seconds]
 '-dt': Float that specifies the time step for the DecisionPolicy instance. [Default 0.04 seconds]
 '-iti': Float that specifies the inter trial time for the DecisionPolicy instance. [Default 1 seconds]
 '-tp': Float that specifies the penalty time for the DecisionPolicy instance. [Default 0 seconds]
 '-r' or '--reward': Float that specifies the reward for the DecisionPolicy instance. [Default 10 seconds]
 '-p' or '--penalty': Float that specifies the penalty for the DecisionPolicy instance. [Default 10 seconds]
 '-sf' or '--suffix': A string suffix to paste to the filenames. [Default '']
 
 Example:
 python moving_bounds_fits.py -T 10 -dt 0.001 --save"""
	options =  {'task':1,'ntasks':1,'method':'full','optimizer':'cma','save':False,'plot':False,'fit':True,'time_units':'seconds',
				'T':None,'iti':None,'tp':None,'dt':None,'reward':1,'penalty':0,'n':101,'suffix':''}
	
	expecting_key = True
	key = None
	for i,arg in enumerate(sys.argv[1:]):
		if expecting_key:
			if arg=='-t' or arg=='--task':
				key = 'task'
				expecting_key = False
			elif arg=='-nt' or arg=='--ntasks':
				key = 'ntasks'
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
			elif arg=='-n':
				key = 'n'
				expecting_key = False
			elif arg=='-T':
				key = 'T'
				expecting_key = False
			elif arg=='-dt':
				key = 'dt'
				expecting_key = False
			elif arg=='-iti':
				key = 'iti'
				expecting_key = False
			elif arg=='-tp':
				key = 'tp'
				expecting_key = False
			elif arg=='-r' or arg=='--reward':
				key = 'reward'
				expecting_key = False
			elif arg=='-p' or arg=='--penalty':
				key = 'penalty'
				expecting_key = False
			elif arg=='-sf' or arg=='--suffix':
				key = 'suffix'
				expecting_key = False
			elif arg=='-h' or arg=='--help':
				print script_help
				sys.exit()
			else:
				raise RuntimeError("Unknown option: {opt} encountered in position {pos}. Refer to the help to see the list of options".format(opt=arg,pos=i+1))
		else:
			expecting_key = True
			if key in ['n','task','ntasks']:
				options[key] = int(arg)
			elif key in ['T','dt','iti','tp','reward','penalty']:
				options[key] = float(arg)
			else:
				options[key] = arg
	# Shift task from 1 base to 0 based
	options['task']-=1
	if options['time_units'] not in ['seconds','milliseconds']:
		raise ValueError("Unknown supplied units: '{units}'. Available values are seconds and milliseconds".format(units=options['time_units']))
	if options['method'] not in ['full','confidence_only','full_confidence']:
		raise ValueError("Unknown supplied method: '{method}'. Available values are full, confidence_only and full_confidence".format(method=options['method']))
	return options

if __name__=="__main__":
	options = parse_input()
	method = options['method']
	save = options['save']
	task = options['task']
	ntasks = options['ntasks']
	if save:
		if task==0 and ntasks==1:
			fname = "inference_fit_{method}{suffix}".format(method=method,suffix=options['suffix'])
		else:
			fname = "inference_fit_{method}_{task}_{ntasks}{suffix}".format(method=method,task=task,ntasks=ntasks,suffix=options['suffix'])
		if os.path.isdir("../../figs"):
			fname = "../../figs/"+fname
		if loc==Location.cluster:
			fname+='.png'
			save_object = fname
		else:
			fname+='.pdf'
			save_object = PdfPages(fname)
	else:
		save_object = None
	
	set_time_units(options['time_units'])
	subjects = io.unique_subjects(data_dir)
	subjects.append(io.merge_subjects(subjects))
	for i,s in enumerate(subjects):
		if (i-task)%ntasks==0:
			if options['fit']:
				if method!='confidence_only':
					fit_output = fit(s,method=method,time_units=options['time_units'],n=options['n'],T=options['T'],dt=options['dt'],iti=options['iti'],tp=options['tp'],reward=options['reward'],penalty=options['penalty'],suffix=options['suffix'],optimizer=options['optimizer'])
					f = open("fits/inference_fit_{method}_subject_{id}{suffix}.pkl".format(method=method,id=s.id,suffix=options['suffix']),'w')
					pickle.dump({'fit_output':fit_output,'options':options},f,pickle.HIGHEST_PROTOCOL)
					f.close()
					if method=='full' or method=='two_step':
						fit_output = fit(s,method='confidence_only',time_units=options['time_units'],n=options['n'],T=options['T'],dt=options['dt'],iti=options['iti'],tp=options['tp'],reward=options['reward'],penalty=options['penalty'],suffix=options['suffix'],fixed_parameters=fit_output[0],optimizer=options['optimizer'])
						f = open("fits/inference_fit_confidence_only_subject_{id}{suffix}.pkl".format(id=s.id,suffix=options['suffix']),'w')
						pickle.dump({'fit_output':fit_output,'options':options},f,pickle.HIGHEST_PROTOCOL)
						f.close()
				else:
					f = open("fits/inference_fit_full_subject_{id}{suffix}.pkl".format(id=s.id,suffix=options['suffix']),'r')
					out = pickle.load(f)
					f.close()
					fit_output = fit(s,method='confidence_only',time_units=options['time_units'],n=options['n'],T=options['T'],dt=options['dt'],iti=options['iti'],tp=options['tp'],reward=options['reward'],penalty=options['penalty'],suffix=options['suffix'],fixed_parameters=out['fit_output'][0],optimizer=options['optimizer'])
					f = open("fits/inference_fit_confidence_only_subject_{id}{suffix}.pkl".format(id=s.id,suffix=options['suffix']),'w')
					pickle.dump({'fit_output':fit_output,'options':options},f,pickle.HIGHEST_PROTOCOL)
					f.close()
			if options['plot'] or save:
				plot_fit(s,method=method,save=save_object,display=options['plot'],suffix=options['suffix'])
	if save and not isinstance(save,str):
		save_object.close()
