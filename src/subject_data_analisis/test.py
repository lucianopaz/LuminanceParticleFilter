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
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import utils

subjects = io.filter_subjects_list(io.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment')
all_rt_high = {'2AFC':[],'Auditivo':[],'Luminancia':[]}
all_rt_low = {'2AFC':[],'Auditivo':[],'Luminancia':[]}
all_model_rt_high = {'2AFC':{'t':[],'pdf':[]},'Auditivo':{'t':[],'pdf':[]},'Luminancia':{'t':[],'pdf':[]}}
all_model_rt_low = {'2AFC':{'t':[],'pdf':[]},'Auditivo':{'t':[],'pdf':[]},'Luminancia':{'t':[],'pdf':[]}}
edges = {'2AFC':np.linspace(0,6,51),'Auditivo':np.linspace(0,6,51),'Luminancia':np.linspace(0,1,51)}
for s in subjects:
	fitter_fname = fits.Fitter_filename(experiment=s.experiment,method='full_confidence',name=s.get_name(),
				session=s.get_session(),optimizer='cma',suffix='')
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
	normalization = (np.nansum(rt_high)+np.nansum(rt_low))*dt*len(subjects)
	all_rt_high[s.experiment].append(rt_high/normalization)
	all_rt_low[s.experiment].append(rt_low/normalization)
	
	binary_confidence_pdf,t = fitter.theoretical_rt_confidence_distribution(binary_confidence=True)
	binary_confidence_pdf = np.sum(binary_confidence_pdf,axis=0)/float(len(subjects))
	all_model_rt_high[s.experiment]['t'].append(t)
	all_model_rt_low[s.experiment]['t'].append(t)
	all_model_rt_high[s.experiment]['pdf'].append(binary_confidence_pdf[1])
	all_model_rt_low[s.experiment]['pdf'].append(binary_confidence_pdf[0])
	if s.get_name()!='0':
		break

plt.figure()
for i,k in enumerate(all_rt_high.keys()):
	temp = [len(t) for t in all_model_rt_high[s.experiment]['t']]
	print(temp)
	tlen = max([len(t) for t in all_model_rt_high[s.experiment]['t']])
	dt = all_model_rt_high[s.experiment]['t'][0][1]-all_model_rt_high[s.experiment]['t'][0][0]
	print(dt)
	t = np.arange(tlen,dtype=np.float)*dt
	model_high = np.zeros_like(t)
	model_low = np.zeros_like(t)
	for h,l in zip(all_model_rt_high[s.experiment]['pdf'],all_model_rt_low[s.experiment]['pdf']):
		model_high[:len(h)]+=h
		model_low[:len(l)]+=l
	
	rt_centers = np.array([0.5*(e1+e0) for e1,e0 in zip(edges[k][1:],edges[k][:-1])])
	plt.subplot(1,3,i)
	all_rt_high[k] = np.array(all_rt_high[k])
	all_rt_low[k] = np.array(all_rt_low[k])
	plt.step(rt_centers,np.nansum(all_rt_high[k],axis=0),'forestgreen')
	plt.step(rt_centers,np.nansum(all_rt_low[k],axis=0),'mediumpurple')
	plt.plot(t,model_high,'forestgreen',linewidth=3)
	plt.plot(t,model_low,'mediumpurple',linewidth=3)
	plt.set_xlim()
	plt.title(k)
plt.show(True)
