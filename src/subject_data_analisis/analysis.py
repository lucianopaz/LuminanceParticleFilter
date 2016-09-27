from __future__ import division
from __future__ import print_function

import numpy as np
import data_io_cognition as io
import cost_time as ct
import fits_cognition as fits
from fits_cognition import Fitter
import matplotlib as mt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os, re, pickle, warnings, json, logging, copy, scipy.integrate, itertools
from sklearn import cluster

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
				key = '_'.join(['experiment_'+subjectSession.experiment,'subject_'+str(un),'session_'+str(us)])
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
	
	key = 'experiment_'+fitter.experiment+'subject_'+fitter.subjectSession.get_name()+'_session_'+fitter.subjectSession.get_session()
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
		try:
			names.append(int(vals['name']))
		except:
			names.append(vals['name'])
		try:
			sessions.append(int(vals['session']))
		except:
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

def scatter_parameters(parameters,parameter_names,names,sessions,experiments,merge=None,merge_method=np.mean):
	unames,indnames = np.unique(names,return_inverse=True)
	uexps,indexps = np.unique(experiments,return_inverse=True)
	usess,indsess = np.unique(sessions,return_inverse=True)
	if not merge is None:
		if merge=='subjects':
			temp_pars = []
			temp_sess = []
			temp_exps = []
			for e in uexps:
				for s in usess:
					inds = np.logical_and(experiments==e,sessions==s)
					if any(inds):
						temp_pars.append(merge_method(parameters[inds],axis=0))
						temp_sess.append(s)
						temp_exps.append(e)
			names = None
			parameters = np.array(temp_pars)
			sessions = np.array(temp_sess)
			experiments = np.array(temp_exps)
			colors = ['r','g','b']
			cbar_im = np.array([[1,0,0],[0,0.5,0],[0,0,1]])
			cbar_labels = ['2AFC','Auditivo','Luminancia']
			categories,ind_categories = np.unique(sessions,return_inverse=True)
			markers = ['o','s','D']
			labels = ['Session 1','Session 2','Session 3']
		elif merge=='sessions':
			temp_pars = []
			temp_nams = []
			temp_exps = []
			for e in uexps:
				for n in unames:
					inds = np.logical_and(experiments==e,names==n)
					if any(inds):
						temp_pars.append(merge_method(parameters[inds],axis=0))
						temp_nams.append(n)
						temp_exps.append(e)
			sessions = None
			parameters = np.array(temp_pars)
			names = np.array(temp_nams)
			experiments = np.array(temp_exps)
			unames,indnames = np.unique(names,return_inverse=True)
			colors = [plt.get_cmap('rainbow')(x) for x in indnames.astype(np.float)/float(len(unames)-1)]
			cbar_im = np.array([plt.get_cmap('rainbow')(x) for x in np.arange(len(unames),dtype=np.float)/float(len(unames)-1)])
			cbar_labels = [str(n) for n in unames]
			categories,ind_categories = np.unique(experiments,return_inverse=True)
			markers = ['o','s','D']
			labels = ['2AFC','Auditivo','Luminancia']
		else:
			raise ValueError('Unknown merge option: {0}'.format(merge))
	else:
		a = np.array([sessions,experiments]).T
		b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
		categories, ind_categories = np.unique(b, return_inverse=True)
		colors = [plt.get_cmap('rainbow')(x) for x in indnames.astype(np.float)/float(len(unames)-1)]
		cbar_im = np.array([plt.get_cmap('rainbow')(x) for x in np.arange(len(unames),dtype=np.float)/float(len(unames)-1)])
		cbar_labels = [str(n) for n in unames]
		markers = ['o','+','s','^','v','8','D','*']
		labels = []
		for i,c in enumerate(categories):
			inds = ind_categories==i
			session = sessions[inds][0]
			experiment = experiments[inds][0]
			labels.append('Exp = '+str(experiment)+' Ses = '+str(session))
	
	decision_inds = np.array([i for i,pn in enumerate(parameter_names) if pn in ['cost','internal_var','phase_out_prob']],dtype=np.intp)
	confidence_inds = np.array([i for i,pn in enumerate(parameter_names) if pn in ['high_confidence_threshold','confidence_map_slope']],dtype=np.intp)
	
	plt.figure(figsize=(10,8))
	gs1 = gridspec.GridSpec(2, 2, left=0.05, right=0.85)
	gs2 = gridspec.GridSpec(1, 1, left=0.90, right=0.93)
	ax1 = plt.subplot(gs1[0])
	ax2 = plt.subplot(gs1[1])
	ax3 = plt.subplot(gs1[2])
	ax4 = plt.subplot(gs1[3])
	for cat_index,category in enumerate(categories):
		inds = ind_categories==cat_index
		ax1.scatter(parameters[inds,decision_inds[0]],parameters[inds,decision_inds[1]],c=colors,marker=markers[cat_index],cmap='rainbow')
		ax2.scatter(parameters[inds,decision_inds[0]],parameters[inds,decision_inds[2]],c=colors,marker=markers[cat_index],cmap='rainbow')
		ax3.scatter(parameters[inds,decision_inds[1]],parameters[inds,decision_inds[2]],c=colors,marker=markers[cat_index],cmap='rainbow')
		ax4.scatter(parameters[inds,confidence_inds[0]],parameters[inds,confidence_inds[1]],c=colors,marker=markers[cat_index],cmap='rainbow',label=labels[cat_index])
	ax1.set_xlabel(parameter_names[decision_inds[0]])
	ax1.set_ylabel(parameter_names[decision_inds[1]])
	ax2.set_xlabel(parameter_names[decision_inds[0]])
	ax2.set_ylabel(parameter_names[decision_inds[2]])
	ax3.set_xlabel(parameter_names[decision_inds[1]])
	ax3.set_ylabel(parameter_names[decision_inds[2]])
	ax4.legend(loc='upper center', fancybox=True, framealpha=0.5, scatterpoints=3)
	ax4.set_xlabel(parameter_names[confidence_inds[0]])
	ax4.set_ylabel(parameter_names[confidence_inds[1]])
	
	ax_cbar = plt.subplot(gs2[0])
	plt.imshow(cbar_im.reshape((-1,1,cbar_im.shape[1])),aspect='auto',cmap=None,interpolation='none',origin='lower',extent=[0,1,0.5,len(cbar_labels)+0.5])
	ax_cbar.xaxis.set_ticks([])
	ax_cbar.yaxis.set_ticks(np.arange(len(cbar_labels))+1)
	ax_cbar.yaxis.set_ticklabels(cbar_labels)
	ax_cbar.tick_params(labelleft=False, labelright=True)
	
	plt.show(True)

def hierarchical_clustering(parameters):
	#~ agg = cluster.AgglomerativeClustering(n_clusters=8, affinity='euclidean', compute_full_tree=True, linkage='ward')
	#~ agg.fit(parameters)
	#~ print(agg)
	#~ print(agg.labels_)
	#~ print(agg.n_leaves_)
	#~ print(agg.n_components_)
	#~ print(agg.children_)
	children,n_components,n_leaves,parents,distances = cluster.ward_tree(parameters, return_distance=True)
	virtual_tree_node_iterator = itertools.count(n_leaves)
	tree = [{'node_id': next(virtual_tree_node_iterator), 'left': x[0], 'right':x[1]} for i,x in enumerate(children)]
	print(children)
	print(n_components)
	print(n_leaves)
	print(parents)
	print(distances)
	print(tree)

if __name__=="__main__":
	summary = get_summary()
	parameters,parameter_names,names,sessions,experiments = get_parameter_array_from_summary(summary)
	scatter_parameters(parameters,parameter_names,names,sessions,experiments)
	#~ scatter_parameters(parameters,parameter_names,names,sessions,experiments,merge='subjects')
	#~ scatter_parameters(parameters,parameter_names,names,sessions,experiments,merge='subjects',merge_method=np.median)
	#~ scatter_parameters(parameters,parameter_names,names,sessions,experiments,merge='sessions')
	#~ scatter_parameters(parameters,parameter_names,names,sessions,experiments,merge='sessions',merge_method=np.median)
	hierarchical_clustering(parameters)
