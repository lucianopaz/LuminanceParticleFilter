from __future__ import division
from __future__ import print_function

import numpy as np
import data_io_cognition as io
import cost_time as ct
import fits_cognition as fits
from fits_cognition import Fitter
import matplotlib as mt
from matplotlib import pyplot as plt
import os, re, pickle, warnings, json, logging, copy, scipy.integrate

def subjectSession_measures(subjectSession):
	data = subjectSession.load_data()
	unique_names = np.unique(data[:,-2].astype(np.int))
	unique_sessions = np.unique(data[:,-1].astype(np.int))
	out = {}
	for un in unique_names:
		for us in unique_sessions:
			inds = np.logical_and(data[:,-2]==un,data[:,-1]==us)
			if any(inds):
				n = np.sum(inds.astype(np.int))
				means = np.mean(data[inds,1:4],axis=0)
				stds = np.std(data[inds,1:4],axis=0)/np.sqrt(float(n))
				auc = io.compute_auc(io.compute_roc(data[inds,2],data[inds,3]))
				hit = data[:,2]==1
				miss = data[:,2]==0
				if any(hit):
					inds1 = np.logical_and(hit,inds)
					hit_means = np.mean(data[inds1,1:4:2],axis=0)
					hit_stds = np.std(data[inds1,1:4:2],axis=0)/np.sqrt(np.sum(inds1).astype(np.float))
				else:
					hit_means = np.nan*np.ones(2)
					hit_stds = np.nan*np.ones(2)
				if any(miss):
					inds1 = np.logical_and(miss,inds)
					miss_means = np.mean(data[inds1,1:4:2],axis=0)
					miss_stds = np.std(data[inds1,1:4:2],axis=0)/np.sqrt(np.sum(inds1).astype(np.float))
				else:
					miss_means = np.nan*np.ones(2)
					miss_stds = np.nan*np.ones(2)
				key = '_'.join(['subject_'+str(un),'session_'+str(us)])
				out[key] = {'experiment':subjectSession.experiment,'n':n,'auc':auc,\
							'name':un,'session':us,\
							'means':{'rt':means[0],'performance':means[1],'confidence':means[2],\
									'hit_rt':hit_means[0],'miss_rt':miss_means[0],\
									'hit_confidence':hit_means[1],'miss_confidence':miss_means[1]},\
							'stds':{'rt':stds[0],'performance':stds[1],'confidence':stds[2],\
									'hit_rt':hit_stds[0],'miss_rt':miss_stds[0],\
									'hit_confidence':hit_stds[1],'miss_confidence':miss_stds[1]}}
	return out

def fitter_measures(fitter):
	parameters = fitter.get_parameters_dict_from_fit_output()
	merit = fitter.forced_compute_full_confidence_merit(parameters)
	model_prediction,t_array = fitter.theoretical_rt_confidence_distribution()
	dt = fitter.dp.dt
	c_array = np.linspace(0,1,fitter.confidence_partition)
	norm = np.sum(model_prediction)
	norm0 = np.sum(model_prediction[0])
	norm1 = np.sum(model_prediction[1])
	performance = norm0/norm
	confidence = np.sum(np.sum(model_prediction,axis=0)*c_array[:,None])/norm
	hit_confidence = np.sum(model_prediction[0]*c_array[:,None])/norm0
	miss_confidence = np.sum(model_prediction[1]*c_array[:,None])/norm1
	rt = np.sum(np.sum(model_prediction,axis=0)*t_array[None,:])/norm
	hit_rt = np.sum(model_prediction[0]*t_array[None,:])/norm0
	miss_rt = np.sum(model_prediction[1]*t_array[None,:])/norm1
	
	pconf_hit = np.hstack((np.zeros(1),np.cumsum(np.sum(model_prediction[0],axis=1)*dt)))
	pconf_hit/=pconf_hit[-1]
	pconf_miss = np.hstack((np.zeros(1),np.cumsum(np.sum(model_prediction[1],axis=1)*dt)))
	pconf_miss/=pconf_miss[-1]
	
	auc = scipy.integrate.trapz(pconf_miss,pconf_hit)
	
	key = 'subject_'+fitter.subjectSession.get_name()+'_session_'+fitter.subjectSession.get_session()
	out = {key:{'experiment':fitter.experiment,'parameters':parameters,'merit':merit,\
				'name':fitter.subjectSession.get_name(),'session':fitter.subjectSession.get_session(),\
				'performance':performance,'rt':rt,'hit_rt':hit_rt,'miss_rt':miss_rt,\
				'confidence':confidence,'hit_confidence':hit_confidence,\
				'miss_confidence':miss_confidence,'auc':auc}}
	return out

def get_summary(method = 'full_confidence', optimizer = 'cma', suffix = '', override=False):
	if override or not(os.path.exists('summary_statistics.pkl') and os.path.isfile('summary_statistics.pkl')):
		subjects = io.filter_subjects_list(io.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment')
		summary = {'experimental':{},'theoretical':{}}
		for s in subjects:
			print(s.get_key())
			fname = fits.Fitter_filename(s.experiment,method,s.get_name(),s.get_session(),optimizer,suffix)
			if not(os.path.exists(fname) and os.path.isfile(fname)):
				continue
			s_measures = subjectSession_measures(s)
			summary['experimental'].update(s_measures)
			fitter = fits.load_Fitter_from_file(fname)
			f_measures = fitter_measures(fitter)
			summary['theoretical'].update(f_measures)
		f = open('summary_statistics.pkl','w')
		pickle.dump(summary,f)
		f.close()
	else:
		f = open('summary_statistics.pkl','r')
		summary = pickle.load(f)
		f.close()
	return summary

def get_parameter_array_from_summary(summary):
	parameter_names = set([])
	parameter_dicts = []
	experiments = []
	sessions = []
	names = []
	for k in summary['theoretical'].keys():
		vals = summary['theoretical'][k]
		names.append(vals['name'])
		sessions.append(vals['session'])
		experiments.append(vals['experiment'])
		parameter_names = parameter_names | set(vals['parameters'].keys())
		parameter_dicts.append(vals['parameters'])
	parameter_names = sorted(list(parameter_names))
	parameters = []
	for pd in parameter_dicts:
		vals = []
		for pn in parameter_names:
			vals.append(pd[pn])
		parameters.append(np.array(vals))
	parameters = np.array(parameters)
	names = np.array(names)
	sessions = np.array(sessions)
	experiments = np.array(experiments)
	ind = [i for i,v in enumerate(parameter_names) if v=='internal_var'][0]
	parameters[:,ind] = normalize_internal_vars(parameters[:,ind],experiments)
	return parameters,parameter_names,names,sessions,experiments

def normalize_internal_vars(internal_vars,experiments):
	unique_experiments = sorted(list(set(experiments)))
	for ue in unique_experiments:
		inds = experiments==ue
		internal_vars[inds] = internal_vars[inds]/np.std(internal_vars[inds],keepdims=True)
	return internal_vars

if __name__=="__main__":
	summary = get_summary()
	parameters,parameter_names,names,sessions,experiments = get_parameter_array_from_summary(summary)
	uses,indses = np.unique(sessions,return_inverse=True)
	uexp,indexp = np.unique(experiments,return_inverse=True)
	c = (indses+indexp*len(uses)).astype(np.float)/float(len(uses)*len(uexp))
	
	decision_inds = np.array([i for i,pn in enumerate(parameter_names) if pn in ['cost','internal_var','phase_out_prob']],dtype=np.intp)
	confidence_inds = np.array([i for i,pn in enumerate(parameter_names) if pn in ['high_confidence_threshold','confidence_map_slope']],dtype=np.intp)
	
	plt.figure()
	plt.subplot(131)
	plt.scatter(parameters[:,decision_inds[0]],parameters[:,decision_inds[1]],c=c)
	plt.xlabel(parameter_names[decision_inds[0]])
	plt.ylabel(parameter_names[decision_inds[1]])
	plt.subplot(132)
	plt.scatter(parameters[:,decision_inds[0]],parameters[:,decision_inds[2]],c=c)
	plt.xlabel(parameter_names[decision_inds[0]])
	plt.ylabel(parameter_names[decision_inds[2]])
	plt.subplot(133)
	plt.scatter(parameters[:,decision_inds[1]],parameters[:,decision_inds[2]],c=c)
	plt.xlabel(parameter_names[decision_inds[1]])
	plt.ylabel(parameter_names[decision_inds[2]])
	
	plt.figure()
	plt.scatter(parameters[:,confidence_inds[0]],parameters[:,confidence_inds[1]],c=c)
	plt.xlabel(parameter_names[confidence_inds[0]])
	plt.ylabel(parameter_names[confidence_inds[1]])
	
	plt.show(True)
