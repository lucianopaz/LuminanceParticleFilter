from __future__ import division
from __future__ import print_function

import numpy as np
import data_io_cognition as io
import cost_time as ct
import fits_cognition as fits
from fits_cognition import Fitter
import matplotlib as mt
mt.use("Qt4Agg") # This program works with Qt only
from matplotlib import pyplot as plt
from matplotlib import colors as mt_colors
import matplotlib.gridspec as gridspec
import os, re, pickle, warnings, json, logging, copy, scipy.integrate, itertools, ete3, sys
from sklearn import cluster
from matplotlib import image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import utils

subjects = io.filter_subjects_list(io.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment')
#~ subjects = io.filter_subjects_list(subjects,'experiment_Luminancia')
exp_edges = {'2AFC':np.linspace(0,6,51),'Auditivo':np.linspace(0,6,51),'Luminancia':np.linspace(0,1,51)}
exp_xlim = {'2AFC':[0,6],'Auditivo':[0,6],'Luminancia':[0,1]}
for s in subjects:
	#~ if s.get_name()!='0':
		#~ continue
	#~ fitter_fname = fits.Fitter_filename(experiment=s.experiment,method='full_confidence',name=s.get_name(),
				#~ session=s.get_session(),optimizer='cma',suffix='')
	#~ fitter_fname = fits.Fitter_filename(experiment=s.experiment,method='full_binary_confidence',name=s.get_name(),
				#~ session=s.get_session(),optimizer='Nelder-Mead',suffix='')
	#~ fitter_fname = fits.Fitter_filename(experiment=s.experiment,method='full_binary_confidence',name=s.get_name(),
				#~ session=s.get_session(),optimizer='basinhopping',suffix='',confidence_map_method='belief')
	fitter_fname = fits.Fitter_filename(experiment=s.experiment,method='binary_confidence_only',name=s.get_name(),
				session=s.get_session(),optimizer='basinhopping',suffix='',confidence_map_method='belief')
	f = open(fitter_fname,'r')
	fitter = pickle.load(f)
	f.close()
	print(fitter.default_bounds())
	print(fitter.default_start_point())
	
	fit_output = fitter._fit_output
	print(fit_output)
	fitted_x = fit_output[0]['high_confidence_threshold']
	fitted_merit = fit_output[1]
	
	bounds = fitter._bounds['high_confidence_threshold']
	merits = []
	xs = []
	
	rt = fitter.rt
	perf = fitter.performance
	conf = fitter.confidence
	
	conf_split = np.median(conf)
	low = conf<conf_split
	high = conf>=conf_split
	
	edges = exp_edges[s.experiment]
	xlim = exp_xlim[s.experiment]
	dt = edges[1]-edges[0]
	if any(high):
		rt_high,_ = np.histogram(rt[high],edges,density=True)
	else:
		rt_high = np.nan*np.ones((edges.shape[0]-1))
	if any(low):
		rt_low,_ = np.histogram(rt[low],edges,density=True)
	else:
		rt_low = np.nan*np.ones((edges.shape[0]-1))
	normalization = (np.nansum(rt_high)+np.nansum(rt_low))*dt
	rt_high/=normalization
	rt_low/=normalization
	
	binary_confidence_pdf,t = fitter.theoretical_rt_confidence_distribution(binary_confidence=True)
	fig = plt.figure(figsize=(11,11))
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	ax1.cla()
	ax2.cla()
	best_pdf = {'merit':np.inf,'pdf':None,'x':None}
	for x in np.linspace(bounds[0],bounds[1],100):
		merit = fitter.binary_confidence_only_merit(x)
		parameters = fitter.get_parameters_dict_from_array(x)
		random_rt_likelihood = 0.5*parameters['phase_out_prob']
		phigh = fitter.high_confidence_mapping(parameters['high_confidence_threshold'],np.inf)
		phigh_low = (phigh,1.-phigh)
		dead_time_convolver = fitter.__fit_internals__['dead_time_convolver']
		pdf = None
		for index,drift in enumerate(fitter.mu):
			binary_confidence_pdf = np.sum(fitter.binary_confidence_pdf(fitter.__fit_internals__['first_passage_times'][drift],
																	  parameters,phigh_low=phigh_low,dead_time_convolver=dead_time_convolver),axis=0)
			if pdf is None:
				pdf = binary_confidence_pdf*fitter.mu_prob[index]
			else:
				pdf+= binary_confidence_pdf*fitter.mu_prob[index]
		t = np.arange(0,binary_confidence_pdf.shape[-1])*fitter.dp.dt
		pdf/= (np.sum(pdf)*fitter.dp.dt)
		random_rt_likelihood = np.ones_like(pdf)
		random_rt_likelihood[:,np.logical_or(t<fitter.min_RT,t>fitter.max_RT)] = 0.
		random_rt_likelihood/=(np.sum(random_rt_likelihood)*fitter.dp.dt)
		pdf = pdf*(1.-parameters['phase_out_prob'])+parameters['phase_out_prob']*random_rt_likelihood
		if merit<best_pdf['merit']:
			best_pdf['merit'] = merit
			best_pdf['pdf'] = pdf
			best_pdf['x'] = x
		
		merits.append(merit)
		xs.append(x)
		ax1.cla()
		ax1.plot(xs,merits)
		ax2.cla()
		ax2.step(edges,np.hstack((rt_high,rt_high[-1])),'forestgreen',label='Subject high',where='post')
		ax2.step(edges,np.hstack((rt_low,rt_low[-1])),'mediumpurple',label='Subject low',where='post')
		ax2.plot(t,pdf[1],'forestgreen',linewidth=3,label='Theoretical high')
		ax2.plot(t,pdf[0],'mediumpurple',linewidth=3,label='Theoretical low')
		ax2.legend(loc='best', fancybox=True, framealpha=0.5)
		ax2.set_xlim(xlim)
		fig.canvas.draw()
		plt.waitforbuttonpress(timeout=0.05)
	ax2.cla()
	ax2.step(edges,np.hstack((rt_high,rt_high[-1])),'forestgreen',label='Subject high',where='post')
	ax2.step(edges,np.hstack((rt_low,rt_low[-1])),'mediumpurple',label='Subject low',where='post')
	ax2.plot(t,best_pdf['pdf'][1],'forestgreen',linewidth=3,label='Theoretical high')
	ax2.plot(t,best_pdf['pdf'][0],'mediumpurple',linewidth=3,label='Theoretical low')
	binary_confidence_pdf,t = fitter.theoretical_rt_confidence_distribution(binary_confidence=True)
	binary_confidence_pdf = np.sum(binary_confidence_pdf,axis=0)
	ax2.plot(t,binary_confidence_pdf[1],'b',linewidth=3,label='theoretical_rt_confidence_distribution high')
	ax2.plot(t,binary_confidence_pdf[0],'r',linewidth=3,label='theoretical_rt_confidence_distribution low')
	plt.suptitle('fit results: x={0} f(x)={1}\nManual method: x={2} f(x)={3}'.format(fitted_x,fitted_merit,best_pdf['x'],best_pdf['merit']))
