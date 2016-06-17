from __future__ import division

import enum, os, sys, math, scipy, pickle, cma, copy
import data_io as io
import cost_time as ct
import numpy as np
from utils import normpdf
import matplotlib as mt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import moving_bounds_fits as mo

hand_picked_thresh = [0.8,0.74,0.8,0.7,0.64,0.93,0.81]
subjects = io.unique_subjects(mo.data_dir)
subjects.append(io.merge_subjects(subjects))
fit_files = ["fits/inference_fit_full_subject_{0}.pkl".format(s.id) for s in subjects]

stats = {}
for index,(s,fname) in enumerate(zip(subjects,fit_files)):
	f = open(fname,'r')
	fit = pickle.load(f)
	f.close()
	if isinstance(fit,dict):
		time_units = fit['options']['time_units']
		fit = fit['fit_output']
	else:
		time_units = 'seconds'
	try:
		cost = fit[0]['cost']
		dead_time = fit[0]['dead_time']
		dead_time_sigma = fit[0]['dead_time_sigma']
		phase_out_prob = fit[0]['phase_out_prob']
	except:
		cost = fit[0][0]
		dead_time = fit[0][1]
		dead_time_sigma = fit[0][2]
		phase_out_prob = fit[0][3]
	high_conf_threshold = hand_picked_thresh[index]
	dat,t,d = s.load_data()
	if time_units=='seconds':
		rt = dat[:,1]*1e-3
	else:
		rt = dat[:,1]
	perf = dat[:,2]
	conf = dat[:,3]-1
	
	stats[s.id] = {}
	if time_units=='seconds':
		stats[s.id]['cost'] = cost
		stats[s.id]['dead_time'] = dead_time
		stats[s.id]['dead_time_sigma'] = dead_time_sigma
	else:
		stats[s.id]['cost'] = cost*1e3
		stats[s.id]['dead_time'] = dead_time*1e3
		stats[s.id]['dead_time_sigma'] = dead_time_sigma*1e3
	stats[s.id]['phase_out_prob'] = phase_out_prob
	stats[s.id]['high_conf_threshold'] = high_conf_threshold
	stats[s.id]['n'] = dat.shape[0]
	stats[s.id]['fitted_blocks'] = len(np.unique(dat[:,-1]))
	stats[s.id]['max_rt'] = np.max(rt)
	stats[s.id]['mean_rt'] = np.mean(rt)
	stats[s.id]['median_rt'] = np.median(rt)
	stats[s.id]['mean_perf'] = np.mean(perf)
	stats[s.id]['mean_conf'] = np.mean(conf)
	stats[s.id]['std_rt'] = np.std(rt)
	stats[s.id]['std_perf'] = np.std(perf)/np.sqrt(len(perf))
	stats[s.id]['std_conf'] = np.std(conf)/np.sqrt(len(conf))
	stats[s.id]['mean_high_hit_rt'] = np.mean(rt[np.logical_and(perf==1,conf==1)])
	stats[s.id]['mean_low_hit_rt'] = np.mean(rt[np.logical_and(perf==1,conf==0)])
	stats[s.id]['mean_high_miss_rt'] = np.mean(rt[np.logical_and(perf==0,conf==1)])
	stats[s.id]['mean_low_miss_rt'] = np.mean(rt[np.logical_and(perf==0,conf==0)])
	stats[s.id]['std_high_hit_rt'] = np.std(rt[np.logical_and(perf==1,conf==1)])
	stats[s.id]['std_low_hit_rt'] = np.std(rt[np.logical_and(perf==1,conf==0)])
	stats[s.id]['std_high_miss_rt'] = np.std(rt[np.logical_and(perf==0,conf==1)])
	stats[s.id]['std_low_miss_rt'] = np.std(rt[np.logical_and(perf==0,conf==0)])
	print 'Subject: ',s.id
	print 'Parameters:'
	print 'cost = ({0})1/s, dead_time = {1:1.3f}s, dead_time_sigma = {2:1.3f}s, phase_out_prob = {3:1.3f}, high_conf_threshold = {4}'.format(stats[s.id]['cost'],stats[s.id]['dead_time'],stats[s.id]['dead_time_sigma'],stats[s.id]['phase_out_prob'],stats[s.id]['high_conf_threshold'])
	print 'Subject stats:'
	print 'n = {0}, fitted_blocks = {1}, rt = ({2:1.3f}+-{3:1.3f})s, perf = {4:1.3f}+-{5:1.3f}, conf = {6:1.3f}+-{7:1.3f}'.format(stats[s.id]['n'],stats[s.id]['fitted_blocks'],stats[s.id]['mean_rt'],stats[s.id]['std_rt'],stats[s.id]['mean_perf'],stats[s.id]['std_perf'],stats[s.id]['mean_conf'],stats[s.id]['std_conf'])
	print 'high hit rt = ({0:1.3f}+-{1:1.3f})s, low hit rt = ({2:1.3f}+-{3:1.3f})s, high miss rt = ({4:1.3f}+-{5:1.3f})s, low miss rt = ({6:1.3f}+-{7:1.3f})s'.format(stats[s.id]['mean_high_hit_rt'],stats[s.id]['std_high_hit_rt'],stats[s.id]['mean_low_hit_rt'],stats[s.id]['std_low_hit_rt'],stats[s.id]['mean_high_miss_rt'],stats[s.id]['std_high_miss_rt'],stats[s.id]['mean_low_miss_rt'],stats[s.id]['std_low_miss_rt'])
	print '-----------------------------------'
f = open('fits/stats_inference_full.pkl','w')
pickle.dump(stats,f)
f.close()
