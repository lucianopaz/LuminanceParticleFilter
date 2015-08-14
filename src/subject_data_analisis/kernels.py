#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package for computing psychophysical kernels """

import numpy as np
import itertools, math, sys, warnings

def pad_stim(stim,length,axis=1,pad=np.nan,start=0):
	"""
	Takes the stimulus data and returns a numpy.ndarray with an axis
	padded with the desired value up to the desired axis length.
	
	Syntax:
	padded_stim = pad_stim(stim,length,axis=1,pad=np.nan,start=0)
	
	Input:
	- stim: An iterable. Each element must be a numpy.ndarray holding
	  the trial's stimulus. Each element can have different shapes. This
	  function converts all the stimulus data of different shapes into a
	  single ndarray with a shape equal to (Ntrials,...,length,...),
	  where Ntrials is the number of elements in the stim iterable, the
	  ellipses '...' mean that the stim element's shape is copied except
	  for the specified axis that is set to length
	- length: An integer. The length of the padded axis
	- axis: The axis that will be padded and whose length will be set to
	  'length'. Axis cannot be set to 0.
	- pad: The value used to pad the stimulus data
	- start: An integer. The starting index where the stimulus will be
	  copied
	
	Example:
	import numpy as np
	stim = np.random.random((5,2,2))
	print pad_stim(stim,10)
	print pad_stim(stim,10,start=2)
	
	"""
	if axis==0:
		raise(ValueError("axis cannot be set to 0. axis 0 is the trials axis and it cannot be changed"))
	padded_stim = []
	for s in stim:
		shape_s = list(s.shape)
		shape_temp = list(s.shape)
		shape_temp[axis-1] = int(length)
		temp = pad*np.ones(tuple(shape_temp))
		slices = [range(sh) for sh in shape_temp]
		slices[axis-1] = range(start,min(int(length),shape_s[axis-1]+int(start)))
		temp[np.ix_(*slices)] = s.copy()
		padded_stim.append(temp)
	return np.array(padded_stim)

def kernels(fluctuations,selection,confidence,is_binary_confidence=True,locked_on_onset=True,RT_ind=None):
	"""
	Compute psychophysical decision and confidence kernels, locked to
	stimulus onset or to response time.
	
	Syntax: 
	decision_kernel,confidence_kernel,decision_kernel_std,confidence_kernel_std = 
	    kernels(fluctuations,selection,confidence,is_binary_confidence=True,locked_on_onset=True,RT_ind=None)
	
	Input:
	- fluctuations must be a numpy.array or ndarray with shape (Ntrials, Selection index, Time index)
	  with the stimulus fluctuation data for each trial and selectable option
	- selection must be an iterable with Ntrials elements that indicate
	  the selection index for each trial
	- confidence must have the same shape as selection. It can either
	  have binary values indicating high (1) or low confidence (0), or a
	  float indicating the probability of high confidence reports
	- is_binary_confidence is a bool value that flags whether confidence
	  holds binary or real values. Default is true.
	- locked_on_onset is a bool value indicating if the kernels are to
	  be computed locked on stimulus onset (taken to be the first time
	  step). Default is true.
	- RT_ind is mandatory if locked_on_onset is False. Must be an
	  iterable with Ntrials elements that indicate the time step at 
	  which the subject responded.
	
	Output:
	- decision_kernel, decision_kernel_std: 2xM numpy arrays containing
	  the selected and non selected patch fluctuations of the decision
	  kernel (and standad deviation).
	- confidence_kernel, confidence_kernel_std: 2xM numpy arrays
	  analogous to the decision_kernel but containg the information on
	  the confidence kernel.
	"""
	
	fluct_shape = list(fluctuations.shape)
	if not locked_on_onset:
		# If the locked on onset kernels are wanted, the fluctuations are
		# shifted on a new vector so that RT_ind is located at its center
		fluct_shape[2] = fluct_shape[2]*2-1
		fluct_T_dec = np.nan*np.ones(tuple(fluct_shape))
		
		for i,f in enumerate(fluctuations):
			fluct_T_dec[i,:,fluctuations.shape[2]-1-RT_ind[i]:fluct_shape[2]-RT_ind[i]] = f.copy()
		fluctuations = fluct_T_dec
	fluct_shape = tuple(fluct_shape)
	
	# High and low, selected and non selected fluctuation vectors
	high = np.nan*np.ones(fluct_shape)
	low = np.nan*np.ones(fluct_shape)
	high_trials = 0
	low_trials = 0
	for trial,f in enumerate(fluctuations):
		if np.isnan(selection[trial]):
			continue
		if is_binary_confidence:
			if confidence[trial]==1:
				high[trial,0] = f[selection[trial]]
				high[trial,1] = f[1-selection[trial]]
				high_trials+=1
			else:
				low[trial,0] = f[selection[trial]]
				low[trial,1] = f[1-selection[trial]]
				low_trials+=1
		else:
			high[trial,0] = f[selection[trial]]*confidence[trial]
			high[trial,1] = f[1-selection[trial]]*confidence[trial]
			low[trial,0] = f[selection[trial]]*(1.-confidence[trial])
			low[trial,1] = f[1-selection[trial]]*(1.-confidence[trial])
			high_trials+=1
			low_trials+=1
	if not locked_on_onset:
		# Crop the shifted time steps with less than half the trials'
		# information.
		# The horrible logical operation reads as follows:
		# For every time index, if the number of trials with not nan
		# values is less than half the total number of valid trials,
		# then discard all data with that time index by setting its
		# value to nan.
		high[:,:,np.sum(np.logical_not(np.isnan(high[:,0,:])),axis=0)<0.5*high_trials] = np.nan
		low[:,:,np.sum(np.logical_not(np.isnan(low[:,0,:])),axis=0)<0.5*high_trials] = np.nan
	
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)
		decision_kernel = np.nanmean(np.concatenate((high,low),axis=0),axis=0)
		confidence_kernel = np.nanmean(high,axis=0)-np.nanmean(low,axis=0)
		
		decision_kernel_std = np.nanstd(np.concatenate((high,low),axis=0),axis=0)/np.sqrt(fluct_shape[0])
		confidence_kernel_std = np.nanstd(high,axis=0)/np.sqrt(high_trials)+np.nanstd(low,axis=0)/np.sqrt(low_trials)
	return decision_kernel,confidence_kernel,decision_kernel_std,confidence_kernel_std

def test(data_dir='/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles'):
	"""
	Compute luminance experiment control subjects' decision and
	confidence kernels. Provide data_dir fullpath if the default path is
	not correct
	"""
	try:
		from matplotlib import pyplot as plt
		loaded_plot_libs = True
	except:
		loaded_plot_libs = False
	import data_io as dio
	# Load subject data
	subjects = dio.unique_subjects(data_dir)
	ms = dio.merge_subjects(subjects)
	dat,t,d = ms.load_data()
	# Average the 4 bar luminances and substract the mean luminance
	t = (np.mean(t,axis=2,keepdims=False).T-dat[:,0]).T
	d = np.mean(d,axis=2,keepdims=False)-50
	# Join target and distractor data into a single variable, fluctuations
	fluctuations = np.transpose(np.array([t,d]),(1,0,2))
	# Pad the data with nans to be able to compute kernels locked on response time
	max_rt = max(dat[:,1])
	ISI = 40.
	padding_length = int(math.ceil(max_rt/ISI))
	if max_rt/ISI==padding_length:
		padding_length+=1
	RT_ind = np.floor_divide(dat[:,1],ISI).astype(int)
	RT_ind[RT_ind==padding_length] = padding_length-1
	fluctuations = pad_stim(fluctuations,padding_length,axis=2)
	
	# Compute kernels locked on stimulus onset
	sdk,sck,sdk_std,sck_std = kernels(fluctuations,1-dat[:,2],dat[:,3]-1)
	sT = np.array(range(padding_length),dtype=float)*ISI
	
	# Compute kernels locked on response time
	rdk,rck,rdk_std,rck_std = kernels(fluctuations,1-dat[:,2],dat[:,3]-1,locked_on_onset=False,RT_ind=RT_ind)
	rT = (np.array(range(2*padding_length-1),dtype=float)-padding_length+1)*ISI
	
	if loaded_plot_libs:
		# Plot kernels
		plt.figure(figsize=(13,10))
		# Decision kernel locked on stimulus onset
		plt.subplot(221)
		plt.plot(sT,sdk[0],color='b')
		plt.plot(sT,sdk[1],color='r')
		plt.fill_between(sT,sdk[0]-sdk_std[0],sdk[0]+sdk_std[0],color='b',alpha=0.3,edgecolor=None)
		plt.fill_between(sT,sdk[1]-sdk_std[1],sdk[1]+sdk_std[1],color='r',alpha=0.3,edgecolor=None)
		plt.plot(plt.gca().get_xlim(),[0,0],color='k')
		plt.ylabel('Fluctuation [$cd/m^{2}$]')
		plt.legend(['$D_{S}$','$D_{N}$'])
		plt.title('Locked on onset')
		# Confidence kernel locked on stimulus onset
		ax = plt.subplot(223,sharex=plt.subplot(221),sharey=plt.subplot(221))
		plt.plot(sT,sck[0],color='b')
		plt.plot(sT,sck[1],color='r')
		plt.fill_between(sT,sck[0]-sck_std[0],sck[0]+sck_std[0],color='b',alpha=0.3,edgecolor=None)
		plt.fill_between(sT,sck[1]-sck_std[1],sck[1]+sck_std[1],color='r',alpha=0.3,edgecolor=None)
		plt.plot(plt.gca().get_xlim(),[0,0],color='k')
		plt.xlabel('Time [ms]')
		plt.ylabel('Fluctuation [$cd/m^{2}$]')
		plt.legend(['$C_{S}$','$C_{N}$'])
		# Decision kernel locked on response time
		plt.subplot(222)
		plt.plot(rT,rdk[0],color='b')
		plt.plot(rT,rdk[1],color='r')
		plt.fill_between(rT,rdk[0]-rdk_std[0],rdk[0]+rdk_std[0],color='b',alpha=0.3,edgecolor=None)
		plt.fill_between(rT,rdk[1]-rdk_std[1],rdk[1]+rdk_std[1],color='r',alpha=0.3,edgecolor=None)
		plt.plot(plt.gca().get_xlim(),[0,0],color='k')
		plt.legend(['$D_{S}$','$D_{N}$'])
		plt.title('Locked on response time')
		# Confidence kernel locked on response time
		plt.subplot(224,sharex=plt.subplot(222),sharey=plt.subplot(222))
		plt.plot(rT,rck[0],color='b')
		plt.plot(rT,rck[1],color='r')
		plt.fill_between(rT,rck[0]-rck_std[0],rck[0]+rck_std[0],color='b',alpha=0.3,edgecolor=None)
		plt.fill_between(rT,rck[1]-rck_std[1],rck[1]+rck_std[1],color='r',alpha=0.3,edgecolor=None)
		plt.plot(plt.gca().get_xlim(),[0,0],color='k')
		plt.xlabel('Time - RT [ms]')
		plt.legend(['$C_{S}$','$C_{N}$'])
		plt.show()
	else:
		print sdk,sck,sdk_std,sck_std
		print rdk,rck,rdk_std,rck_std
	return 0

if __name__=="__main__":
	if len(sys.argv)>1:
		test(sys.argv[1])
	else:
		test()
