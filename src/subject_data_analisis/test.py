from __future__ import division
from __future__ import print_function

import numpy as np
import data_io_cognition as io
import cost_time as ct
import fits_cognition as fits
from fits_cognition import Fitter
import matplotlib as mt
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
all_rt_high = {'2AFC':[],'Auditivo':[],'Luminancia':[]}
all_rt_low = {'2AFC':[],'Auditivo':[],'Luminancia':[]}
all_model_rt_high = {'2AFC':{'t':[],'pdf':[]},'Auditivo':{'t':[],'pdf':[]},'Luminancia':{'t':[],'pdf':[]}}
all_model_rt_low = {'2AFC':{'t':[],'pdf':[]},'Auditivo':{'t':[],'pdf':[]},'Luminancia':{'t':[],'pdf':[]}}
n = {'2AFC':0,'Auditivo':0,'Luminancia':0}
edges = {'2AFC':np.linspace(0,6,51),'Auditivo':np.linspace(0,6,51),'Luminancia':np.linspace(0,1,51)}
for s in subjects:
	#~ if s.get_name()!='12':
		#~ continue
	#~ fitter_fname = fits.Fitter_filename(experiment=s.experiment,method='full_confidence',name=s.get_name(),
				#~ session=s.get_session(),optimizer='cma',suffix='')
	#~ fitter_fname = fits.Fitter_filename(experiment=s.experiment,method='full_binary_confidence',name=s.get_name(),
				#~ session=s.get_session(),optimizer='Nelder-Mead',suffix='')
	fitter_fname = fits.Fitter_filename(experiment=s.experiment,method='full_binary_confidence',name=s.get_name(),
				session=s.get_session(),optimizer='basinhopping',suffix='',confidence_map_method='belief')
	#~ fitter_fname = fits.Fitter_filename(experiment=s.experiment,method='binary_confidence_only',name=s.get_name(),
				#~ session=s.get_session(),optimizer='basinhopping',suffix='',confidence_map_method='belief')
	f = open(fitter_fname,'r')
	fitter = pickle.load(f)
	f.close()
	print(fitter.get_key())
	rt = fitter.rt
	perf = fitter.performance
	conf = fitter.confidence
	
	conf_split = np.median(conf)
	low = conf<conf_split
	high = conf>=conf_split
	
	dt = edges[s.experiment][1]-edges[s.experiment][0]
	if any(high):
		rt_high,_ = np.histogram(rt[high],edges[s.experiment],density=True)
	else:
		rt_high = np.nan*np.ones((edges[s.experiment].shape[0]-1))
	if any(low):
		rt_low,_ = np.histogram(rt[low],edges[s.experiment],density=True)
	else:
		rt_low = np.nan*np.ones((edges[s.experiment].shape[0]-1))
	normalization = (np.nansum(rt_high)+np.nansum(rt_low))*dt
	all_rt_high[s.experiment].append(rt_high/normalization*float(len(perf)))
	all_rt_low[s.experiment].append(rt_low/normalization*float(len(perf)))
	
	binary_confidence_pdf,t = fitter.theoretical_rt_confidence_distribution(binary_confidence=True)
	binary_confidence_pdf = np.sum(binary_confidence_pdf,axis=0)
	all_model_rt_high[s.experiment]['t'].append(t)
	all_model_rt_low[s.experiment]['t'].append(t)
	all_model_rt_high[s.experiment]['pdf'].append(binary_confidence_pdf[1]*float(len(perf)))
	all_model_rt_low[s.experiment]['pdf'].append(binary_confidence_pdf[0]*float(len(perf)))
	n[s.experiment]+= float(len(perf))
	#~ plt.step(edges[s.experiment],np.hstack((rt_high/normalization,(rt_high/normalization)[-1])),'forestgreen',label='Subject high',where='post')
	#~ plt.step(edges[s.experiment],np.hstack((rt_low/normalization,(rt_low/normalization)[-1])),'mediumpurple',label='Subject low',where='post')
	#~ plt.plot(t,binary_confidence_pdf[1],'forestgreen',linewidth=3,label='Theoretical high')
	#~ plt.plot(t,binary_confidence_pdf[0],'mediumpurple',linewidth=3,label='Theoretical low')
	#~ plt.show(True)

plt.figure()
experiment_alias = {'2AFC':'Contrast','Auditivo':'Auditory','Luminancia':'Luminance'}
row_index = {'2AFC':1,'Auditivo':2,'Luminancia':3}
xlim = {'2AFC':[0,6],'Auditivo':[0,6],'Luminancia':[0,1]}
plot_gs = gridspec.GridSpec(3, 1,left=0.25, right=0.98)
exp_scheme_gs = gridspec.GridSpec(3, 1,left=0., right=0.16,hspace=0.25)
for i,k in enumerate(all_rt_high.keys()):
	temp = [len(t) for t in all_model_rt_high[k]['t']]
	tlen = max([len(t) for t in all_model_rt_high[k]['t']])
	dt = all_model_rt_high[k]['t'][0][1]-all_model_rt_high[k]['t'][0][0]
	t = np.arange(tlen,dtype=np.float)*dt
	model_high = np.zeros_like(t)
	model_low = np.zeros_like(t)
	for h,l in zip(all_model_rt_high[k]['pdf'],all_model_rt_low[k]['pdf']):
		model_high[:len(h)]+=h
		model_low[:len(l)]+=l
	
	rt_centers = np.array([0.5*(e1+e0) for e1,e0 in zip(edges[k][1:],edges[k][:-1])])
	ax = plt.subplot(plot_gs[row_index[k]-1,0])
	all_rt_high[k] = np.nansum(np.array(all_rt_high[k])/n[k],axis=0)
	all_rt_low[k] = np.nansum(np.array(all_rt_low[k])/n[k],axis=0)
	plt.step(edges[k],np.hstack((all_rt_high[k],all_rt_high[k][-1])),'forestgreen',label='Subject high',where='post')
	plt.step(edges[k],np.hstack((all_rt_low[k],all_rt_low[k][-1])),'mediumpurple',label='Subject low',where='post')
	plt.plot(t,model_high/n[k],'forestgreen',linewidth=3,label='Theoretical high')
	plt.plot(t,model_low/n[k],'mediumpurple',linewidth=3,label='Theoretical low')
	ax.set_xlim(xlim[k])
	if row_index[k]==1:
		plt.legend(loc='best', fancybox=True, framealpha=0.5)
	elif row_index[k]==3:
		plt.xlabel('RT [s]')
	
	#~ ax = plt.subplot(plot_gs[row_index[k]-1,1])
	#~ high_ratio = all_rt_high[k]/(all_rt_high[k]+all_rt_low[k])
	#~ high_ratio_std = np.sqrt(high_ratio*(1-high_ratio)/(all_rt_high[k]+all_rt_low[k])/n[k])
	#~ high_ratio_std[high_ratio_std<1e-2] = 1e-2
	#~ plt.plot(rt_centers,high_ratio,color='b',label='Subject high')
	#~ plt.fill_between(rt_centers,high_ratio+high_ratio_std,high_ratio-high_ratio_std,color='b',alpha=0.4,label='Subject high ratio')
	#~ plt.plot(t,model_high/(model_high+model_low),'r',linewidth=2,label='Theoretical high ratio')
	#~ ax.set_xlim(xlim[k])
	#~ if row_index[k]==1:
		#~ plt.legend(loc='best', fancybox=True, framealpha=0.5)
	#~ elif row_index[k]==3:
		#~ plt.xlabel('RT [s]')
	
	aximg = plt.subplot(exp_scheme_gs[row_index[k]-1])
	exp_scheme = mpimg.imread('../../figs/'+k+'.png')
	aximg.imshow(exp_scheme)
	aximg.set_axis_off()
	plt.title(experiment_alias[k])
plt.show(True)
