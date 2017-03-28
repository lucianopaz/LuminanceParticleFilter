#-*- coding: UTF-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import data_io_cognition as io
import cost_time as ct
import fits_cognition as fits
from fits_cognition import Fitter
import utils
import matplotlib as mt
from matplotlib import pyplot as plt
from matplotlib import colors as mt_colors
import matplotlib.gridspec as gridspec
import os, re, pickle, warnings, json, logging, copy, scipy.integrate, itertools, ete3, sys
from sklearn import cluster
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
try:
	from diptest.diptest import dip
except:
	from utils import dip

def default_tree_layout(node):
	"""
	default_tree_layout(node)
	
	Takes an ete3.TreeNode instance and sets its default style adding the
	appropriate TextFaces
	
	"""
	style = ete3.NodeStyle(vt_line_width=3,hz_line_width=3,size=0)
	if node.is_leaf():
		portions = node.name.split('_')
		experiment = None
		subject = None
		session = None
		name_alias = []
		if not portions[0] in ['subj','ses']:
			start_ind = 1
			experiment = portions[0]
		else:
			start_ind = 0
		for key,val in zip(portions[start_ind::2],portions[start_ind+1::2]):
			if key=='subj':
				subject = val
			elif key=='ses':
				session = val
		if not experiment is None:
			bgcolor = {'2AFC':'#FF0000','Auditivo':'#008000','Luminancia':'#0000FF'}[experiment]
			fgcolor = {'2AFC':'#000000','Auditivo':'#000000','Luminancia':'#000000'}[experiment]
		else:
			bgcolor = {'1':'#FF0000','2':'#008000','3':'#0000FF'}[session]
			fgcolor = {'1':'#000000','2':'#000000','3':'#000000'}[session]
		if experiment:
			name_alias.append({'2AFC':'Con','Auditivo':'Aud','Luminancia':'Lum'}[experiment])
		if subject:
			name_alias.append('Subj {0}'.format(subject))
		if session:
			name_alias.append('Ses {0}'.format(session))
		name_alias = ' '.join(name_alias)
		style['vt_line_color'] = bgcolor
		style['hz_line_color'] = bgcolor
		style['size'] = 3
		style['fgcolor'] = bgcolor
		face = ete3.TextFace(name_alias,fgcolor=fgcolor)
		#~ face.rotation = -15
		node.add_face(face, column=0, position="aligned")
	else:
		child_leaf_color = None
		equal_leaf_types = True
		for child_leaf in (n for n in node.iter_descendants("postorder") if n.is_leaf()):
			portions = child_leaf.name.split('_')
			try:
				bgcolor = {'2AFC':'#FF0000','Auditivo':'#008000','Luminancia':'#0000FF'}[portions[0]]
			except:
				bgcolor = {'1':'#FF0000','2':'#008000','3':'#0000FF'}[portions[-1]]
			if child_leaf_color is None:
				child_leaf_color = bgcolor
			elif child_leaf_color!=bgcolor:
				equal_leaf_types = False
				break
		if equal_leaf_types:
			style['vt_line_color'] = child_leaf_color
			style['hz_line_color'] = child_leaf_color
	node.set_style(style)

def default_tree_style(mode='r',title=None):
	"""
	default_tree_style(mode='r')
	
	mode can be 'r' or 'c'. Returns an ete3.TreeStyle instance with a
	rectangular or circular display depending on the supplied mode
	
	"""
	tree_style = ete3.TreeStyle()
	tree_style.layout_fn = default_tree_layout
	tree_style.show_leaf_name = False
	tree_style.show_scale = False
	if not title is None:
		tree_style.title.add_face(ete3.TextFace(title, fsize=18), column=0)
	if mode=='r':
		tree_style.rotation = 90
		tree_style.branch_vertical_margin = 10
	else:
		tree_style.mode = 'c'
		tree_style.arc_start = 0 # 0 degrees = 3 o'clock
		tree_style.arc_span = 180
	return tree_style

class Analyzer():
	def __init__(self,method = 'full_confidence', optimizer = 'cma', suffix = '',
				cmap_meth='log_odds', fits_path='fits_cognition/',
				override=False,n_clusters=2, affinity='euclidean',
				linkage='ward', pooling_func=np.nanmean, connectivity=None):
		"""
		Analyzer(method = 'full_confidence', optimizer = 'cma', suffix = '',
				cmap_meth='log_odds', fits_path='fits_cognition/',
				override=False,n_clusters=2, affinity='euclidean',
				linkage='ward', pooling_func=np.nanmean, connectivity=None)
		
		This Class implements an interface between the
		data_io_cognition.SubjectSession and the fits_cognition.Fitter
		that constructs a summary of relevant statistics and features
		of the experimental data and theoretical predictions, and can
		then perform relevant analyces and graphics from them.
		
		The main analysis is the hierarchical clustering of the fitted
		parameters, which is performed with the scikit learn (sklearn)
		package's AgglomerativeClustering class. Thus, when constructing
		the Analyzer, many input parameters are simply parameters used
		for the creation of an AgglomerativeClustering instance.
		
		Input:
			method: Fitter instance's method.
			optimizer: Fitter instance's optimizer.
			suffix: Fitter instance's suffix.
			cmap_meth: Fitter intance's high_confidence_mapping_method.
			
			The four parameters listed above are used to determine the
			Fitter files that should be loaded to get the relevant
			statistics.
			
			override: Bool by default False. The construction of the
				experimental and theoretical statistics takes a lot of
				time. To improve succesive runtimes, the Analyzer, will
				attempt to load the summary statistics from the file
				'summary_{method}_{optimizer}_{cmapmeth}{suffix}.pkl',
				and if it does not exist, it will perform the
				computations and then save the summaries to the above
				mentioned file. If the override parameter is True, the
				summary statistics are not loaded from the file,
				they are computed and the contents of
				'summary_{method}_{optimizer}_{cmapmeth}{suffix}.pkl'
				are overriden with the currently computed statistics.
			
			The following parameters are used to construct the 
			sklearn.cluster.AgglomerativeClustering instance. Refer to
			the scikit learn documentation for a detailed description
			of each of these parameters:
			n_clusters: Int. Default 2
			affinity: Str. Default 'euclidean'
			linkage: Str. Default 'ward'
			pooling_func: Callable. Default numpy.nanmean
			connectivity: Optional array-like or callable. Default None.
			
			We will only mention that the pooling_func is also used to
			pool together the parameters when applying merges in the
			scatter_parameters, controlled_scatter and cluster methods.
			Furthermore, the affinity and linkage are used to compute
			the cluster span.
		
		"""
		self.method = method
		self.optimizer = optimizer
		self.suffix = suffix
		self.cmap_meth = cmap_meth
		self.fits_path = fits_path
		self.get_summary(override=override)
		self.init_clusterer(n_clusters=n_clusters, affinity=affinity,\
				linkage=linkage, pooling_func=pooling_func, connectivity=connectivity)
	
	def init_clusterer(self,n_clusters=2, affinity='euclidean', compute_full_tree=True,
						linkage='ward', pooling_func=np.nanmean,connectivity=None):
		"""
		self.init_clusterer(n_clusters=2, affinity='euclidean', compute_full_tree=True,
						linkage='ward', pooling_func=np.nanmean,connectivity=None)
		
		This method constructs a sklearn.cluster.AgglomerativeClustering
		instance using the call:
		sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters,
			affinity=affinity, compute_full_tree=compute_full_tree,
			linkage=linkage, pooling_func=pooling_func,connectivity=connectivity)
		
		The parameters used for the above mentioned construction are the
		input parameters supplied to this function.
		
		"""
		self.agg_clusterer = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity,\
				connectivity=connectivity,compute_full_tree=compute_full_tree, linkage=linkage,\
				pooling_func=pooling_func)
		self.linkage = linkage
		self.affinity = affinity
		self.pooling_func=pooling_func
		self.spanner = self.get_cluster_spanner()
	
	def set_pooling_func(self,pooling_func):
		"""
		self.set_pooling_func(pooling_func)
		
		Set the pooling function used for parameter clustering.
		
		"""
		self.agg_clusterer.set_params(**{'pooling_func':pooling_func})
		self.pooling_func = pooling_func
		self.spanner = self.get_cluster_spanner()
	
	def set_linkage(self,linkage):
		"""
		self.set_linkage(linkage)
		
		Set the linkage used for parameter clustering.
		
		"""
		self.agg_clusterer.set_params(**{'linkage':linkage})
		self.linkage = linkage
		self.spanner = self.get_cluster_spanner()
	
	def set_affinity(self,affinity):
		"""
		self.set_affinity(affinity)
		
		Set the affinity used for parameter clustering.
		
		"""
		self.agg_clusterer.set_params(**{'affinity':affinity})
		self.affinity = affinity
		self.spanner = self.get_cluster_spanner()
	
	def get_cluster_spanner(self):
		"""
		spanner = self.get_cluster_spanner()
		
		Get a callable that computes a given cluster's span. To compute
		a cluster's span, call
		spanner(cluster)
		
		The cluster must be a 2D numpy array, where the axis=0 holds
		separate cluster members and the axis=1 holds the different
		variables.
		
		"""
		if self.linkage=='ward':
			if self.affinity=='euclidean':
				spanner = lambda x:np.sum((x-self.pooling_func(x,axis=0))**2)
			elif self.affinity in ['l1','l2','manhattan','cosine']:
				raise ValueError('Ward linkage only accepts euclidean affinity. However, affinity attribute was set to {0}.'.format(self.affinity))
			else:
				raise AttributeError('Unknown affinity attribute value {0}.'.format(self.affinity))
		elif self.linkage=='complete':
			if self.affinity=='euclidean':
				spanner = lambda x:np.max(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
			elif self.affinity=='l1' or self.affinity=='manhattan':
				spanner = lambda x:np.max(np.sum(np.abs(x[:,None,:]-x[None,:,:]),axis=2))
			elif self.affinity=='l2':
				spanner = lambda x:np.max(np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2)))
			elif self.affinity=='cosine':
				spanner = lambda x:np.max(np.sum((x[:,None,:]*x[None,:,:]))/(np.sqrt(np.sum(x[:,None,:]*x[:,None,:],axis=2,keepdims=True))*np.sqrt(np.sum(x[None,:,:]*x[None,:,:],axis=2,keepdims=True))))
			else:
				raise AttributeError('Unknown affinity attribute value {0}.'.format(self.affinity))
		elif self.linkage=='average':
			if self.affinity=='euclidean':
				spanner = lambda x:np.mean(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
			elif self.affinity=='l1' or self.affinity=='manhattan':
				spanner = lambda x:np.mean(np.sum(np.abs(x[:,None,:]-x[None,:,:]),axis=2))
			elif self.affinity=='l2':
				spanner = lambda x:np.mean(np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2)))
			elif self.affinity=='cosine':
				spanner = lambda x:np.mean(np.sum((x[:,None,:]*x[None,:,:]))/(np.sqrt(np.sum(x[:,None,:]*x[:,None,:],axis=2,keepdims=True))*np.sqrt(np.sum(x[None,:,:]*x[None,:,:],axis=2,keepdims=True))))
			else:
				raise AttributeError('Unknown affinity attribute value {0}.'.format(self.affinity))
		else:
			raise AttributeError('Unknown linkage attribute value {0}.'.format(self.linkage))
		return spanner
	
	def get_summary_filename(self):
		"""
		self.get_summary_filename()
		
		Simply returns 'summary_{method}_{optimizer}_{cmap_meth}{suffix}.pkl'
		formated with the Analyzer instance's attribute values.
		
		"""
		return 'summary_{method}_{optimizer}_{cmap_meth}{suffix}.pkl'.format(
				method=self.method,optimizer=self.optimizer,suffix=self.suffix,
				cmap_meth=self.cmap_meth)
	
	def get_summary(self,override=False):
		"""
		self.get_summary(override=False)
		
		Get the experimental and theoretical summaries. The input
		'override', if True, signals that the method must compute the
		individual summaries and then override the Analyzer's summary
		filename. If False, this method first tries to load the
		summaries from the Analyzer's summary filename, and if said file
		does not exist, it computes the summaries and then saves them.
		
		Output:
			summary: A dictionary with keys 'experimental' and 'theoretical'.
				Each key holds the output of the methods
				self.subjectSession_measures ('experimental' key) and
				self.fitter_measures ('theoretical' key), called for
				every subject and session that complies with
				data_io_cognition.filter_subjects_list(data_io_cognition.unique_subject_sessions(fits_cognition.raw_data_dir),'all_sessions_by_experiment')
		
		"""
		if override or not(os.path.exists(self.get_summary_filename()) and os.path.isfile(self.get_summary_filename())):
			subjects = io.filter_subjects_list(io.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment')
			self.summary = {'experimental':{},'theoretical':{}}
			for s in subjects:
				print(s.get_key())
				fname = fits.Fitter_filename(experiment=s.experiment,
											 method=self.method,
											 name=s.get_name(),
											 session=s.get_session(),
											 optimizer=self.optimizer,
											 suffix=self.suffix,
											 confidence_map_method=self.cmap_meth,
											 fits_path=self.fits_path)
				if not(os.path.exists(fname) and os.path.isfile(fname)):
					continue
				s_measures = self.subjectSession_measures(s)
				self.summary['experimental'].update(s_measures)
				fitter = fits.load_Fitter_from_file(fname)
				f_measures = self.fitter_measures(fitter)
				self.summary['theoretical'].update(f_measures)
			f = open(self.get_summary_filename(),'w')
			pickle.dump(self.summary,f)
			f.close()
		else:
			f = open(self.get_summary_filename(),'r')
			self.summary = pickle.load(f)
			f.close()
		return self.summary
	
	def subjectSession_measures(self,subjectSession):
		"""
		self.subjectSession_measures(subjectSession)
		
		Get the summary statistics of a SubjectSession instance.
		
		Output:
			stats: A dict with keys equal to
				'experiment_{experiment}_subject_{name}_session_{session}'
				where experiment, name and session are the unique tuple
				values encountered in the subjectSession data.
				The value assigned to each key is itself a dict with shape:
				{'experiment':...,'n':...,'auc':...,
				'name':...,'session':...,
				'means':{'rt':...,'performance':...,'confidence':...,\
						'hit_rt':...,'miss_rt':...,\
						'hit_confidence':...,'miss_confidence':...},
				'stds':{'rt':...,'performance':...,'confidence':...,\
						'hit_rt':...,'miss_rt':...,\
						'hit_confidence':...,'miss_confidence':...},
				'medians':{'rt':...,'confidence':...,\
						'hit_rt':...,'miss_rt':...,\
						'hit_confidence':...,'miss_confidence':...}}
		
		"""
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
					medians = np.median(data[inds,1:4],axis=0)
					stds = np.std(data[inds,1:4],axis=0)/np.sqrt(float(n))
					auc = io.compute_auc(io.compute_roc(data[inds,2],data[inds,3]))
					hit = data[:,2]==1
					miss = data[:,2]==0
					if any(hit):
						inds1 = np.logical_and(hit,inds)
						hit_means = np.mean(data[inds1,1:4:2],axis=0)
						hit_medians = np.median(data[inds1,1:4:2],axis=0)
						hit_stds = np.std(data[inds1,1:4:2],axis=0)/np.sqrt(np.sum(inds1).astype(np.float))
					else:
						hit_means = np.nan*np.ones(2)
						hit_medians = np.nan*np.ones(2)
						hit_stds = np.nan*np.ones(2)
					if any(miss):
						inds1 = np.logical_and(miss,inds)
						miss_means = np.mean(data[inds1,1:4:2],axis=0)
						miss_medians = np.median(data[inds1,1:4:2],axis=0)
						miss_stds = np.std(data[inds1,1:4:2],axis=0)/np.sqrt(np.sum(inds1).astype(np.float))
					else:
						miss_means = np.nan*np.ones(2)
						miss_medians = np.nan*np.ones(2)
						miss_stds = np.nan*np.ones(2)
					dipval = dip(data[inds,3], min_is_0=True, x_is_sorted=False)
					key = '_'.join(['experiment_'+subjectSession.experiment,'subject_'+str(un),'session_'+str(us)])
					out[key] = {'experiment':subjectSession.experiment,'n':n,'auc':auc,
								'name':un,'session':us,'multi_mod_index':dipval,
								'means':{'rt':means[0],'performance':means[1],'confidence':means[2],
										'hit_rt':hit_means[0],'miss_rt':miss_means[0],
										'hit_confidence':hit_means[1],'miss_confidence':miss_means[1]},
								'stds':{'rt':stds[0],'performance':stds[1],'confidence':stds[2],
										'hit_rt':hit_stds[0],'miss_rt':miss_stds[0],
										'hit_confidence':hit_stds[1],'miss_confidence':miss_stds[1]},
								'medians':{'rt':medians[0],'confidence':medians[2],
										'hit_rt':hit_medians[0],'miss_rt':miss_medians[0],
										'hit_confidence':hit_medians[1],'miss_confidence':miss_medians[1]}}
		return out
	
	def fitter_measures(self,fitter):
		"""
		self.fitter_measures(fitter)
		
		Get the summary statistics of a Fitter instance.
		
		Output:
			stats: A dict with keys equal to
				'experiment_{experiment}_subject_{name}_session_{session}'
				where experiment, name and session are the Fitter
				instance's experiment attribute,
				fitter.subjectSession.get_name() and
				fitter.subjectSession.get_session() returned values.
				The value assigned to each key is itself a dict with shape:
				{'experiment':...,'parameters':...,'full_merit':...,
				'full_confidence_merit':...,'confidence_only_merit':...,
				'name':...,'session':...,
				'performance':...,'rt':...,'hit_rt':...,'miss_rt':...,
				'confidence':...,'hit_confidence':...,
				'miss_confidence':...,'auc':...}
		
		"""
		parameters = fitter.get_parameters_dict_from_fit_output()
		full_confidence_merit = fitter.forced_compute_full_confidence_merit(parameters)
		full_merit = fitter.forced_compute_full_merit(parameters)
		confidence_only_merit = fitter.forced_compute_confidence_only_merit(parameters)
		try:
			# First try to load the fitter stats from memory
			f = open(fitter.get_save_file_name().replace('.pkl','_stats.pkl'),'r')
			fitter_stats = pickle.load(f)
			f.close()
		except:
			# If no stats file is found or the load fails, then compute the stats
			fitter_stats = fitter.stats(return_mean_rt=True,return_mean_confidence=True,return_median_rt=True,
					return_median_confidence=True,return_std_rt=True,return_std_confidence=True,
					return_auc=True)
		
		performance = fitter_stats['performance']
		performance_conditioned = fitter_stats['performance_conditioned']
		confidence = fitter_stats['mean_confidence']
		confidence_median = fitter_stats['median_confidence']
		hit_confidence = fitter_stats['mean_confidence_conditioned'][0]
		miss_confidence = fitter_stats['mean_confidence_conditioned'][1]
		hit_confidence_median = fitter_stats['median_confidence_conditioned'][0]
		miss_confidence_median = fitter_stats['median_confidence_conditioned'][1]
		rt = fitter_stats['mean_rt']
		mean_rt_perf = fitter_stats['mean_rt_perf']
		hit_rt = mean_rt_perf[0]
		miss_rt = mean_rt_perf[1]
		rt_median = fitter_stats['median_rt']
		rt_median_perf = fitter_stats['median_rt_perf']
		hit_rt_median = rt_median_perf[0]
		miss_rt_median = rt_median_perf[1]
		
		auc = fitter_stats['auc']
		
		key = 'experiment_'+fitter.experiment+'_subject_'+fitter.subjectSession.get_name()+'_session_'+fitter.subjectSession.get_session()
		out = {key:{'experiment':fitter.experiment,'parameters':parameters,'full_merit':full_merit,
					'full_confidence_merit':full_confidence_merit,'confidence_only_merit':confidence_only_merit,
					'name':fitter.subjectSession.get_name(),'session':fitter.subjectSession.get_session(),
					'performance_mean':performance,'rt_mean':rt,'hit_rt_mean':hit_rt,'miss_rt_mean':miss_rt,
					'confidence_mean':confidence,'hit_confidence_mean':hit_confidence,
					'miss_confidence_mean':miss_confidence,'rt_median':rt_median,
					'hit_rt_median':hit_rt_median,'miss_rt_median':miss_rt_median,
					'confidence_median':confidence_median,'hit_confidence_median':hit_confidence_median,
					'miss_confidence_median':miss_confidence_median}}
		return out
	
	def get_parameter_array_from_summary(self,summary=None,normalize={'internal_var':'experiment'},normalization_function=lambda x: x/np.nanstd(x)):
		"""
		self.get_parameter_array_from_summary(summary=None,normalize={'internal_var':'experiment'},normalization_function=lambda x: x/np.nanstd(x))
		
		Get the array of parameters, experiments, sessions and names
		from the summary['theoretical'] dict.
		
		Input:
			summary: A summary dict or None. If None, the Analyzer's
				summary instance is used instead.
			normalize: None or a dict whos keys are parameter names. If
				None, no normalization is performed. If it is a dict,
				the values indicate the grouping method that will be used
				to normalize the parameter values. Each group is then
				divided by the output of a call to the normalization_function
				on itself. Four methods are available:
				'all': All parameters are taken into a single group.
				'experiment': One group for each experiment is formed.
				'session': One group for each session is formed.
				'name': One group for each name is formed.
				By default normalize is {'internal_var':'experiment'}
				because the internal_var is completely different
				for each experiment.
			normalization_function: A callable that is passed to the
				method self.normalize_parameters in order to
				normalize the Fitter parameters before returning them.
				This function is only used if normalize is not None.
				Refer to self.normalize_parameters for more information.
		
		Output:
			parameters: 2D numpy array. Axis=0 corresponds to different
				experiment, name and subject tuple values, while axis=1
				corresponds to different parameter names.
			parameter_names: A list with the parameter names in the order
				in which they appear in the parameters output
			names: A 1D numpy array with the SubjectSession names that
				correspond to each parameter.
			sessions: A 1D numpy array with the SubjectSession sessions
				that correspond to each parameter.
			experiments: A 1D numpy array with the SubjectSession
				experiments that correspond to each parameter.
		
		"""
		if summary is None:
			summary = self.summary
		parameter_dicts = []
		self._parameter_names = set([])
		self._experiments = []
		self._sessions = []
		self._names = []
		for k in summary['theoretical'].keys():
			vals = summary['theoretical'][k]
			try:
				self._names.append(int(vals['name']))
			except:
				self._names.append(vals['name'])
			try:
				self._sessions.append(int(vals['session']))
			except:
				self._sessions.append(vals['session'])
			self._experiments.append(vals['experiment'])
			self._parameter_names = self._parameter_names | set(vals['parameters'].keys())
			parameter_dicts.append(vals['parameters'])
		self._parameter_names = sorted(list(self._parameter_names))
		self._parameters = []
		for pd in parameter_dicts:
			vals = []
			for pn in self._parameter_names:
				if pn=='high_confidence_threshold':
					if self.cmap_meth=='log_odds':
						val = pd[pn] if pd[pn]<=2. else np.nan
					elif self.cmap_meth=='belief':
						val = pd[pn]+0.7/pd['confidence_map_slope']
						#~ val = pd[pn]
					else:
						val = pd[pn]
				elif pn=='confidence_map_slope':
					if self.cmap_meth=='log_odds':
						val = pd[pn] if pd[pn]<=100. else np.nan
					elif self.cmap_meth=='belief':
						val = pd[pn] if pd[pn]<=40. else np.nan
					else:
						val = pd[pn]
				else:
					val = pd[pn]
				vals.append(val)
			self._parameters.append(np.array(vals))
		self._parameters = np.array(self._parameters)
		self._names = np.array(self._names)
		self._sessions = np.array(self._sessions)
		self._experiments = np.array(self._experiments)
		if normalize:
			group_dict = {'all':None,
						  'experiment':self._experiments,
						  'session':self._sessions,
						  'name':self._names}
			for par in normalize.keys():
				par_ind = self._parameter_names.index(par)
				group = group_dict[normalize[par]]
				self._parameters[:,par_ind] = self.normalize_parameters(parameters=self._parameters[:,par_ind],
									  group=group, normalization_function=normalization_function)
		return self._parameters,self._parameter_names,self._names,self._sessions,self._experiments
	
	def get_summary_stats_array(self,summary=None,normalize={'internal_var':'experiment'},normalization_function=lambda x: x/np.nanstd(x)):
		"""
		self.get_summary_stats_array(summary=None,normalize={'internal_var':'experiment'},normalization_function=lambda x: x/np.nanstd(x))
		
		Get the two numpy arrays of the summary statistics. One for the
		experimental data and another for the theoretical data.
		
		Input:
			summary: A summary dict or None. If None, the Analyzer's
				summary instance is used instead.
			normalize: None or a dict whos keys are parameter names. If
				None, no normalization is performed. If it is a dict,
				the values indicate the grouping method that will be used
				to normalize the parameter values. Each group is then
				divided by the output of a call to the normalization_function
				on itself. Four methods are available:
				'all': All parameters are taken into a single group.
				'experiment': One group for each experiment is formed.
				'session': One group for each session is formed.
				'name': One group for each name is formed.
				By default normalize is {'internal_var':'experiment'}
				because the internal_var is completely different
				for each experiment.
			normalization_function: A callable that is passed to the
				method self.normalize_parameters in order to
				normalize the Fitter parameters before returning them.
				This function is only used if normalize is not None.
				Refer to self.normalize_parameters for more information.
		
		Output:
			(experimental,theoretical): Two numpy arrays with named
				fields. It is posible to access the field names from the
				attribute experimental.dtype.names.
				The two arrays have different field names, except for a
				few. The most important are 'experiment', 'session' and
				'name' which encode the corresponding experiment, session
				and subject name. These fields should be used to get the
				corresponding experimental and theoretical entries.
				
				List of experimental entries:
				- experiment: Experiment
				- name: Subject Name
				- session: Session
				- n: Number of trials
				- auc: Area under to ROC curve
				- multi_mod_index: Hartigan's dip test statistic
				- rt_mean: Mean RT
				- hit_rt_mean: Mean RT for the correct trials
				- miss_rt_mean: Mean RT for the incorrect trials
				- performance_mean: Mean performance
				- confidence_mean: Mean confidence
				- hit_confidence_mean: Mean confidence for the correct trials
				- miss_confidence_mean: Mean confidence for the incorrect trials
				- rt_std: Standard deviation of RT
				- hit_rt_std: Standard deviation of RT for the correct trials
				- miss_rt_std: Standard deviation of RT for the incorrect trials
				- performance_std: Standard deviation of performance
				- confidence_std: Standard deviation of confidence
				- hit_confidence_std: Standard deviation of confidence for the correct trials
				- miss_confidence_std: Standard deviation of confidence for the incorrect trials
				- rt_median: Median RT
				- hit_rt_median: Median RT for the correct trials
				- miss_rt_median: Median RT for the incorrect trials
				- confidence_median: Median confidence
				- hit_confidence_median: Median confidence for the correct trials
				- miss_confidence_median: Median confidence for the incorrect trials
				
				List of theoretical entries:
				- experiment: Experiment
				- name: Subject Name
				- session: Session
				- full_merit: Fitter full_merit value
				- full_confidence_merit: Fitter full_confidence_merit value
				- confidence_only_merit: Fitter confidence_only_merit value
				- auc: Area under to ROC curve
				- rt_mean: Mean RT
				- hit_rt_mean: Mean RT for the correct trials
				- miss_rt_mean: Mean RT for the incorrect trials
				- performance_mean: Mean performance
				- confidence_mean: Mean confidence
				- hit_confidence_mean: Mean confidence for the correct trials
				- miss_confidence_mean: Mean confidence for the incorrect trials
				- rt_median: Median RT
				- hit_rt_median: Median RT for the correct trials
				- miss_rt_median: Median RT for the incorrect trials
				- confidence_median: Median confidence
				- hit_confidence_median: Median confidence for the correct trials
				- miss_confidence_median: Median confidence for the incorrect trials
				- cost: Fitter 'cost' parameter value
				- internal_var: Fitter 'internal_var' parameter value
				- phase_out_prob: Fitter 'phase_out_prob' parameter value
				- dead_time: Fitter 'dead_time' parameter value
				- dead_time_sigma: Fitter 'dead_time_sigma' parameter value
				- high_confidence_threshold: Fitter 'high_confidence_threshold' parameter value
				- confidence_map_slope: Fitter 'confidence_map_slope' parameter value
		
		"""
		if summary is None:
			summary = self.summary
		
		experimental = []
		experimental_ind_names = []
		theoretical = []
		theoretical_ind_names = []
		max_experiment_len = 0
		max_name_len = 0
		parameter_names = []
		for k in summary['theoretical'].keys():
			teo = summary['theoretical'][k]
			exp = summary['experimental'][k]
			for teo_k in teo.keys():
				if teo_k=='experiment':
					max_experiment_len = max([max_experiment_len,len(teo[teo_k])])
				elif teo_k=='name':
					max_name_len = max([max_name_len,len(str(teo[teo_k]))])
				if teo_k!='parameters':
					if teo_k not in theoretical_ind_names:
						theoretical_ind_names.append(teo_k)
						theoretical.append([teo[teo_k]])
					else:
						theoretical[theoretical_ind_names.index(teo_k)].append(teo[teo_k])
				else:
					for par in teo[teo_k].keys():
						par = str(par)
						if par=='high_confidence_threshold':
							if self.cmap_meth=='log_odds':
								val = teo[teo_k][par] if teo[teo_k][par]<=2. else np.nan
							elif self.cmap_meth=='belief':
								val = teo[teo_k][par]+0.7/teo[teo_k]['confidence_map_slope']
								#~ val = teo[teo_k][par]
							else:
								val = teo[teo_k][par]
						elif par=='confidence_map_slope':
							if self.cmap_meth=='log_odds':
								val = teo[teo_k][par] if teo[teo_k][par]<=100. else np.nan
							elif self.cmap_meth=='belief':
								val = teo[teo_k][par] if teo[teo_k][par]<=40. else np.nan
							else:
								val = teo[teo_k][par]
						else:
							val = teo[teo_k][par]
						if par not in parameter_names:
							parameter_names.append(par)
							theoretical_ind_names.append(par)
							theoretical.append([val])
						else:
							theoretical[theoretical_ind_names.index(par)].append(val)
			for exp_k in exp.keys():
				if exp_k=='experiment':
					max_experiment_len = max([max_experiment_len,len(exp[exp_k])])
				elif exp_k=='name':
					max_name_len = max([max_name_len,len(str(exp[exp_k]))])
				if exp_k not in ['means','stds','medians']:
					if exp_k not in experimental_ind_names:
						experimental_ind_names.append(exp_k)
						experimental.append([exp[exp_k]])
					else:
						experimental[experimental_ind_names.index(exp_k)].append(exp[exp_k])
				else:
					for nested_key in exp[exp_k].keys():
						composed_key = nested_key+{'means':'_mean','stds':'_std','medians':'_median'}[exp_k]
						if composed_key not in experimental_ind_names:
							experimental_ind_names.append(composed_key)
							experimental.append([exp[exp_k][nested_key]])
						else:
							experimental[experimental_ind_names.index(composed_key)].append(exp[exp_k][nested_key])
		dtype_dict = {'experiment':'S'+str(max_experiment_len),
					  'n':'i',
					  'session':'i',
					  'name':'S'+str(max_name_len)}
		exp_dtype = []
		for exp_ind in experimental_ind_names:
			try:
				exp_dtype.append((str(exp_ind),dtype_dict[exp_ind]))
			except:
				exp_dtype.append((str(exp_ind),'f'))
		teo_dtype = []
		for teo_ind in theoretical_ind_names:
			try:
				teo_dtype.append((str(teo_ind),dtype_dict[teo_ind]))
			except:
				teo_dtype.append((str(teo_ind),'f'))
		N = len(experimental[0])
		experimental_ = []
		theoretical_ = []
		for i in range(N):
			experimental_.append(tuple([e[i] for e in experimental]))
			theoretical_.append(tuple([t[i] for t in theoretical]))
		experimental = np.array(experimental_,exp_dtype)
		theoretical = np.array(theoretical_,teo_dtype)
		
		if normalize:
			group_dict = {'all':None,
						  'experiment':theoretical['experiment'],
						  'session':theoretical['session'],
						  'name':theoretical['name']}
			for par in normalize.keys():
				group = group_dict[normalize[par]]
				theoretical[par] = self.normalize_parameters(parameters=theoretical[par],
									  group=group, normalization_function=normalization_function)
		return experimental,theoretical
	
	def normalize_parameters(self,parameters,group=None,normalization_function=lambda x: x/np.nanstd(x)):
		"""
		self.normalize_parameters(parameters,group,normalization_function=lambda x: x/np.nanstd(x))
		
		The hierarchical clustering of parameters depends on the distance
		between parameters where each parameter name is interpreted as
		a separate dimension in a certain metric space (Default is
		euclidean but other metrics can be specified in the affinity).
		However, each parameter is usually located in an interval that
		has a completely different scale across experiments and parameter
		names. This function seeks to normalize the parameter values
		to make them inhabit similarly sized regions, and thus improve
		the clustering. The most problematic parameter is the
		'internal_var' because its units vary between experiments.
		To solve this, normalize_parameters allows the user to specify
		different grouping methods for different parameters.
		
		Input:
			parameters: A 1D numpy array with the parameter values for
				a single parameter name.
			group: Can be None or a 1D numpy array with the same
				number of elements as the input 'parameters'. The
				distinct values of 'group' define separate groups
				and the normalization function is called separately
				on each group. For example, the input 'group'
				could be an array with the corresponding experiment of
				each parameter value. Thus, the normalization_function
				would be called once for each experiment on the separate
				subsets of parameters that corresponded to each
				experiment. If None, the normalization_function is
				called on the entire 'parameters' array.
			normalization_function: A callable that is called on each
				group to normalize the parameter values. Default is
				lambda x: x/numpy.nanstd(x). Other normalization_function
				implementations must be able to broadcast numpy arrays,
				receive a single numpy array as input and return another
				numpy array with the same shape as the input array.
				
		
		Output:
			normalized_parameters: A numpy array that is the result of
				calling the normalization_function on each group.
		
		"""
		if group is None:
			return normalization_function(parameters)
		else:
			normalized_parameters = np.empty_like(parameters)
			unique_group = np.unique(group)
			for g in unique_group:
				inds = group==g
				normalized_parameters[inds] = normalization_function(parameters[inds])
			return normalized_parameters
	
	def scatter_parameters(self,merge=None,show=False):
		"""
		self.scatter_parameters(merge=None,show=False)
		
		Plot the parameters 'internal_var', 'cost' and 'phase_out_prob'
		against each other, and also plot 'high_confidence_threshold'
		against 'confidence_map_slope' in a 2x2 subplot with a colorbar
		that represents subject names or experiments depending on the
		'merge' input value.
		
		Input:
			merge: None, 'names' or 'sessions'. If None, all the parameters
				are plotted. If 'names', the parameters that correspond
				to different subject names but to the same experiment
				and session are pooled together. If 'sessions', the same
				procedure is applied but for parameters that correspond
				to different sessions.
			show: A bool that if True, shows the plotted figure and
				freezes the execution until said figure is closed.
		
		"""
		try:
			unames,indnames = np.unique(self._names,return_inverse=True)
		except:
			self.get_parameter_array_from_summary()
			unames,indnames = np.unique(self._names,return_inverse=True)
		uexps,indexps = np.unique(self._experiments,return_inverse=True)
		usess,indsess = np.unique(self._sessions,return_inverse=True)
		if not merge is None:
			if merge=='names':
				temp_pars = []
				temp_sess = []
				temp_exps = []
				for e in uexps:
					for s in usess:
						inds = np.logical_and(self._experiments==e,self._sessions==s)
						if any(inds):
							temp_pars.append(self.pooling_func(self._parameters[inds],axis=0))
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
						inds = np.logical_and(self._experiments==e,self._names==n)
						if any(inds):
							temp_pars.append(self.pooling_func(self._parameters[inds],axis=0))
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
			names = np.copy(self._names)
			sessions = np.copy(self._sessions)
			experiments = np.copy(self._experiments)
			parameters = np.copy(self._parameters)
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
		
		decision_inds = np.array([i for i,pn in enumerate(self._parameter_names) if pn in ['cost','internal_var','phase_out_prob']],dtype=np.intp)
		confidence_inds = np.array([i for i,pn in enumerate(self._parameter_names) if pn in ['high_confidence_threshold','confidence_map_slope']],dtype=np.intp)
		
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
		ax1.set_xlabel(self._parameter_names[decision_inds[0]])
		ax1.set_ylabel(self._parameter_names[decision_inds[1]])
		ax2.set_xlabel(self._parameter_names[decision_inds[0]])
		ax2.set_ylabel(self._parameter_names[decision_inds[2]])
		ax3.set_xlabel(self._parameter_names[decision_inds[1]])
		ax3.set_ylabel(self._parameter_names[decision_inds[2]])
		ax4.legend(loc='upper center', fancybox=True, framealpha=0.5, scatterpoints=3)
		ax4.set_xlabel(self._parameter_names[confidence_inds[0]])
		ax4.set_ylabel(self._parameter_names[confidence_inds[1]])
		
		ax_cbar = plt.subplot(gs2[0])
		plt.imshow(cbar_im.reshape((-1,1,cbar_im.shape[1])),aspect='auto',cmap=None,interpolation='none',origin='lower',extent=[0,1,0.5,len(cbar_labels)+0.5])
		ax_cbar.xaxis.set_ticks([])
		ax_cbar.yaxis.set_ticks(np.arange(len(cbar_labels))+1)
		ax_cbar.yaxis.set_ticklabels(cbar_labels)
		ax_cbar.tick_params(labelleft=False, labelright=True)
		
		if show:
			plt.show(True)
	
	def controlled_scatter(self,scattered_parameters=['cost','internal_var','phase_out_prob'],axes=None,color_category='experiment',marker_category='session',merge=None,show=False):
		"""
		self.controlled_scatter(scattered_parameters=['cost','internal_var','phase_out_prob'],axes=None,color_category='experiment',marker_category='session',merge=None,show=False)
		
		A more refined version of the method scatter_parameters where
		it is posible to select the group of parameters to plot against
		each other, to which axes to plot them along with other options.
		
		Input:
			scattered_parameters: A list of valid Fitter parameter
				names to plot against each other. The list must have 2
				or 3 elements, and the scatter will be in 2D or 3D
				accordingly.
			axes: None or a matplotlib.Axes or mpl_toolkits.mplot3d.Axes3D
				instance. If None, the axes is created on the fly in a
				new figure. If not None, then the supplied axes will be
				used to call the scatter method.
			color_category: Can be 'experiment', 'session' or 'name'. It
				is used to indicate if the maker's color should indicate
				the corresponding experiment, session or subject name.
			marker_category: Can be 'experiment' or 'session'. It is
				used to indicate if the maker should indicate the
				corresponding experiment or session.
			merge: Can be None, 'experiments', 'names' or 'sessions',
				and has the same behavior as the 'merge' input in
				the scatter_parameters method. Refer to the
				scatter_parameters docstring for further details.
			show: A bool that if True, shows the plotted figure and
				freezes the execution until said figure is closed.
		
		Output:
			If show is True, this function returns None.
			If show is False, this function returns the Axes instance
			which was used to scatter the parameters.
		
		"""
		if len(scattered_parameters)<2 or len(scattered_parameters)>3:
			raise ValueError('Can only scatter sets of 2 or 3 parameters. User supplied scattered_parameters={0}'.format(scattered_parameters))
		elif len(scattered_parameters)==2:
			threeD = False
		else:
			threeD = True
		try:
			params = self._parameters
		except:
			self.get_parameter_array_from_summary()
			params = self._parameters
		_x = params[:,self._parameter_names.index(scattered_parameters[0])]
		_y = params[:,self._parameter_names.index(scattered_parameters[1])]
		if threeD:
			_z = params[:,self._parameter_names.index(scattered_parameters[2])]
		else:
			_z = None
		if color_category not in [None,'experiment','session','name']:
			raise ValueError("color_category must be in [None,'experiment','session','name']. User supplied: {0}".format(color_category))
		if marker_category not in [None, 'experiment','session']:
			raise ValueError("marker_category must be in [None,'experiment','session']. User supplied: {0}".format(marker_category))
		if marker_category==color_category and not marker_category is None:
			raise ValueError("If color_category and marker_category are not None, they must be different")
		if merge==(marker_category+'s') or merge==(color_category+'s'):
			raise ValueError("Cannot set merge equal to color_category or marker_category")
		
		if not merge is None:
			unams,indnams = np.unique(self._names,return_inverse=True)
			usess,indsess = np.unique(self._sessions,return_inverse=True)
			uexps,indexps = np.unique(self._experiments,return_inverse=True)
			cat_inds = []
			names = []
			sessions = []
			experiments = []
			if merge=='experiments':
				for i,un in enumerate(unams):
					for us in usess:
						inds = np.logical_and(self._names==un,self._sessions==us)
						if any(inds):
							cat_inds.append(inds)
							names.append(i)
							sessions.append(us)
			elif merge=='sessions':
				for i,un in enumerate(unams):
					for ue in uexps:
						inds = np.logical_and(self._names==un,self._experiments==ue)
						if any(inds):
							cat_inds.append(inds)
							names.append(i)
							experiments.append(ue)
			elif merge=='names':
				for ue in uexps:
					for us in usess:
						inds = np.logical_and(self._experiments==ue,self._sessions==us)
						if any(inds):
							cat_inds.append(inds)
							experiments.append(ue)
							sessions.append(us)
			else:
				raise ValueError("Unknown merge option supplied: {0}. Available options are None, 'experiments', 'sessions' or 'names'".format(merge))
			x = []
			y = []
			if threeD:
				z = []
			for inds in cat_inds:
				x.append(self.pooling_func(_x[inds],axis=0))
				y.append(self.pooling_func(_y[inds],axis=0))
				if threeD:
					z.append(self.pooling_func(_z[inds],axis=0))
			x = np.array(x)
			y = np.array(y)
			if threeD:
				z = np.array(z)
		else:
			unams,names = np.unique(self._names,return_inverse=True)
			sessions = self._sessions
			experiments = self._experiments
			x = _x
			y = _y
			if threeD:
				z = _z
		
		if not color_category is None:
			if color_category=='experiment':
				c = [{'2AFC':[1.,0.,0.],'Auditivo':[0.,0.5,0.],'Luminancia':[0.,0.,1.]}[exp] for exp in experiments]
			elif color_category=='session':
				c = [{'1':[1.,0.,0.],'2':[0.,0.5,0.],'3':[0.,0.,1.]}[ses] for ses in sessions]
			else:
				c = [plt.get_cmap('rainbow')(x) for x in names.astype(np.float)/float(len(unams)-1)]
			c = np.array(c)
		else:
			c = 'k'
			
		if not marker_category is None:
			if marker_category=='experiment':
				uexps,indexps = np.unique(experiments,return_inverse=True)
				marker_inds = [indexps==i for i,ue in enumerate(uexps)]
				markers = [{'2AFC':'o','Auditivo':'s','Luminancia':'D'}[exp] for exp in uexps]
				labels = [{'2AFC':'Con','Auditivo':'Aud','Luminancia':'Lum'}[exp] for exp in uexps]
			else:
				usess,indsess = np.unique(sessions,return_inverse=True)
				marker_inds = [indsess==i for i,ue in enumerate(usess)]
				markers = [{1:'o',2:'s',3:'D'}[int(ses)] for ses in usess]
				labels = [{1:'Ses 1',2:'Ses 2',3:'Ses 3'}[int(ses)] for ses in usess]
		else:
			marker_inds = [np.ones(len(x),dtype=np.bool)]
			markers = ['c']
			labels = [None]
		
		if axes is None:
			fig = plt.figure()
			if threeD:
				axes = fig.add_subplot(111, projection='3d')
			else:
				axes = plt.subplot(111)
		
		parameter_aliases = {'cost':r'$c$',\
							'internal_var':r'$\sigma^{2}$',\
							'phase_out_prob':r'$p_{po}$',\
							'high_confidence_threshold':r'$C_{H}$',\
							'confidence_map_slope':r'$\alpha$',\
							'dead_time':r'$\tau_{c}$',\
							'dead_time_sigma':r'$\sigma_{c}$'}
		if not threeD:
			for m,inds,label in zip(markers,marker_inds,labels):
				try:
					axes.scatter(x[inds],y[inds],c=c[inds],marker=m, s=40, label=label)
				except:
					axes.scatter(x[inds],y[inds],c=c,marker=m, s=40, label=label)
			axes.set_xlabel(parameter_aliases[scattered_parameters[0]],fontsize=16)
			axes.set_ylabel(parameter_aliases[scattered_parameters[1]],fontsize=16)
			axes.set_xticklabels([])
			axes.set_yticklabels([])
		else:
			for m,inds,label in zip(markers,marker_inds,labels):
				try:
					axes.scatter(x[inds],y[inds],z[inds],c=c[inds],marker=m, s=40, label=label)
				except:
					axes.scatter(x[inds],y[inds],z[inds],c=c,marker=m, s=40, label=label)
			axes.set_xlabel(parameter_aliases[scattered_parameters[0]],fontsize=16)
			axes.set_ylabel(parameter_aliases[scattered_parameters[1]],fontsize=16)
			axes.set_zlabel(parameter_aliases[scattered_parameters[2]],fontsize=16)
			axes.view_init(elev=None,azim=145)
			axes.set_xticklabels([])
			axes.set_yticklabels([])
			axes.set_zticklabels([])
		
		if show:
			plt.show(True)
			return None
		else:
			return axes
	
	def cluster(self,merge=None,clustered_parameters=['cost','internal_var','phase_out_prob'],filter_nans='post'):
		"""
		self.cluster(merge=None,clustered_parameters=['cost','internal_var','phase_out_prob'],filter_nans='post')
		
		Method that performs the hierarchical clustering of the Fitter
		parameters.
		
		Input:
			merge: None, or 'experiments', 'sessions', 'names'. This
				input specifies if and how the parameters should be
				merged before being clustered. Initially, there is one
				parameter vector for each individual experiment, session
				and name tuple. If merge is None, all the individual
				parameters are used for the clustering. If 'experiments',
				all the parameters that correspond to the same 'name' and
				'session' pairs are merged (the different experiments
				are pooled together). Analoguously, for 'sessions' and
				'names' the different session and different name values,
				correspondingly, are pooled together.
			clustered_parameters: A list of valid Fitter parameter names.
				If None, all the available Fitter parameter names are
				used. Each parameter name represents a different dimension
				of each parameter array that is passed to the clustering
				fit function.
			filter_nans: A str that can be 'pre', 'post' or 'none' that
				indicates if and how the nans should be handled. If
				'none', no handling is performed. If 'pre', the parameter
				array that holds a nan value in any parameter name, even
				if it is not in the clustered_parameters list is removed.
				If 'post', only the parameters that have a nan value in
				one of the clustered_parameters is removed.
		
		Output:
			tree: An ete3.Tree instance that is built from the Newick
				representation of the
				sklearn.cluster.AgglomerativeClustering.fit() tree
				structure.
		
		"""
		if not filter_nans in ['pre','post','none']:
			raise ValueError('filter_nans must be "pre", "post" or "none". User supplied {0}'.format(filter_nans))
		try:
			parameter_names = self._parameter_names
		except:
			self.get_parameter_array_from_summary()
			parameter_names = self._parameter_names
		clustered_parameters_inds = np.zeros(len(parameter_names),dtype=np.bool)
		if clustered_parameters is None:
			clustered_parameters = parameter_names
		for cpar in clustered_parameters:
			try:
				index = parameter_names.index(cpar)
			except:
				raise ValueError('Clustered parameters must be in {0}. User supplied {1}'.format(parameter_names,cpar))
			clustered_parameters_inds[index] = True
		
		if merge is None:
			X = self._parameters
			leaf_labels = [str(e).strip('\x00')+'_subj_'+str(n).strip('\x00')+'_ses_'+str(s).strip('\x00') for e,n,s in zip(self._experiments,self._names,self._sessions)]
		else:
			uexps,invexps = np.unique(self._experiments,return_inverse=True)
			usess,invsess = np.unique(self._sessions,return_inverse=True)
			unams,invnams = np.unique(self._names,return_inverse=True)
			X = []
			leaf_labels = []
			if merge=='experiments':
				for i,us in enumerate(usess):
					for j,un in enumerate(unams):
						inds = np.logical_and(invsess==i,invnams==j)
						if any(inds):
							X.append(self.pooling_func(self._parameters[inds],axis=0))
							leaf_labels.append('subj_'+str(un)+'_ses_'+str(us))
			elif merge=='sessions':
				for i,ue in enumerate(uexps):
					for j,un in enumerate(unams):
						inds = np.logical_and(invexps==i,invnams==j)
						if any(inds):
							X.append(self.pooling_func(self._parameters[inds],axis=0))
							leaf_labels.append(str(ue)+'_subj_'+str(un))
			elif merge=='names':
				for i,us in enumerate(usess):
					for j,ue in enumerate(uexps):
						inds = np.logical_and(invsess==i,invexps==j)
						if any(inds):
							X.append(self.pooling_func(self._parameters[inds],axis=0))
							leaf_labels.append(str(ue)+'_ses_'+str(us))
			else:
				raise ValueError('Unknown merge option: {0}'.format(merge))
			X = np.array(X)
		# Select X
		if filter_nans=='pre':
			X = X[np.logical_not(np.any(np.isnan(X),axis=1))]
		X = X[:,clustered_parameters_inds]
		if filter_nans=='post':
			X = X[np.logical_not(np.any(np.isnan(X),axis=1))]
		self.agg_clusterer.fit(X)
		newick_tree = self.build_Newick_tree(self.agg_clusterer.children_,self.agg_clusterer.n_leaves_,X,leaf_labels)
		tree = ete3.Tree(newick_tree)
		return tree
	
	def build_Newick_tree(self,children,n_leaves,X,leaf_labels):
		"""
		self.build_Newick_tree(children,n_leaves,X,leaf_labels)
		
		Get a string representation (Newick tree) from the sklearn
		AgglomerativeClustering.fit output.
		
		Input:
			children: AgglomerativeClustering.children_
			n_leaves: AgglomerativeClustering.n_leaves_
			X: parameters supplied to AgglomerativeClustering.fit
			leaf_labels: The label of each parameter array in X
		
		Output:
			ntree: A str with the Newick tree representation
		
		"""
		return self.go_down_tree(children,n_leaves,X,leaf_labels,len(children)+n_leaves-1)[0]+';'
	
	def go_down_tree(self,children,n_leaves,X,leaf_labels,nodename):
		"""
		self.go_down_tree(children,n_leaves,X,leaf_labels,nodename)
		
		Iterative function that traverses the subtree that descends from
		nodename and returns the Newick representation of the subtree.
		
		Input:
			children: AgglomerativeClustering.children_
			n_leaves: AgglomerativeClustering.n_leaves_
			X: parameters supplied to AgglomerativeClustering.fit
			leaf_labels: The label of each parameter array in X
			nodename: An int that is the intermediate node name whos
				children are located in children[nodename-n_leaves].
		
		Output:
			ntree: A str with the Newick tree representation
		
		"""
		nodeindex = nodename-n_leaves
		if nodename<n_leaves:
			return leaf_labels[nodeindex],np.array([X[nodeindex]])
		else:
			node_children = children[nodeindex]
			branch0,branch0samples = self.go_down_tree(children,n_leaves,X,leaf_labels,node_children[0])
			branch1,branch1samples = self.go_down_tree(children,n_leaves,X,leaf_labels,node_children[1])
			node = np.vstack((branch0samples,branch1samples))
			branch0span = self.get_branch_span(branch0samples)
			branch1span = self.get_branch_span(branch1samples)
			nodespan = self.get_branch_span(node)
			branch0distance = nodespan-branch0span
			branch1distance = nodespan-branch1span
			nodename = '({branch0}:{branch0distance},{branch1}:{branch1distance})'.format(branch0=branch0,branch0distance=branch0distance,branch1=branch1,branch1distance=branch1distance)
			return nodename,node
	
	def get_branch_span(self,branchsamples):
		"""
		self.get_branch_span(branchsamples)
		
		Assume that a tree branch is a cluster returned by
		AgglomerativeClustering and compute its corresponding span. The
		returned value will depend on the linkage and affinity.
		
		Output: A float that represents the cluster's span.
		
		"""
		return self.spanner(branchsamples)
	
	def parameter_correlation(self,used_parameters=['cost','internal_var','phase_out_prob','high_confidence_threshold','confidence_map_slope'],
							  method='pearson',nanhandling='pairwise',
							  correct_multiple_comparison_pvalues=True,
							  normalize={'internal_var':'experiment'},
							  normalization_function=lambda x: x/np.nanstd(x)):
		"""
		self.parameter_correlation(used_parameters=['cost','internal_var','phase_out_prob','high_confidence_threshold','confidence_map_slope'],
							  method='pearson',nanhandling='pairwise',
							  correct_multiple_comparison_pvalues=True,
							  normalize={'internal_var':'experiment'},
							  normalization_function=lambda x: x/np.nanstd(x))
		
		Compute the correlation matrix between parameters from the
		parameter values for each experiment, session and subject name.
		
		Input:
			used_parameters: A list with the parameters that will be
				used to compute the correlations.
			method: The method used to compute the correlation matrix.
				Refer to utils.corrcoef for more information.
			nanhandling: The nan handling policy used to compute the
				correlation matrix. Refer to utils.corrcoef for more
				information.
			correct_multiple_comparison_pvalues: A bool. If True, the
				pvalues are corrected for multiple comparisons using the
				Holm-Bonferroni method. If it is False, no correction is
				applied.
			normalize: None or a dict that indicates if and how the
				parameter values should be normalized. Refer to
				self.get_summary_stats_array for more information.
			normalization_function: A callable that is used to normalize
				the parameter values. Refer to
				self.get_summary_stats_array for more information.
		
		Output:
			corrs: A 2D numpy array that holds the correlation values.
				These range from -1 to 1, where 1 indicates full positive
				correlation, -1 indicates full negative correlation and
				0 indicates no correlation. The order in which
				parameters appear in the supplied 'used_parameters'
				controls the order in which they appear in the corrs and
				pvals arrays.
			pvals: A 2D numpy array that holds the correlation statistic's
				p-value. Typically, p-values below 0.05 are considered
				indicative of significant correlation.
		
		"""
		subj,model = self.get_summary_stats_array(normalize=normalize,normalization_function=normalization_function)
		parameters = np.array([model[par] for par in used_parameters])
		corrs,pvals = utils.corrcoef(parameters,method=method,nanhandling=nanhandling)
		if correct_multiple_comparison_pvalues:
			pvals = correct_rho_pval(pvals)
		return corrs,pvals
	
	def correlate_parameters_with_stats(self,used_statistics=['rt_mean','performance_mean','confidence_mean','auc','multi_mod_index'],
							  used_parameters=['cost','internal_var','phase_out_prob','high_confidence_threshold','confidence_map_slope'],
							  method='pearson',nanhandling='pairwise',
							  correct_multiple_comparison_pvalues=True,
							  normalize={'internal_var':'experiment'},
							  normalization_function=lambda x: x/np.nanstd(x)):
		"""
		self.parameter_correlation(used_statistics=['rt_mean','performance_mean','confidence_mean','auc','multi_mod_index'],
							  used_parameters=['cost','internal_var','phase_out_prob','high_confidence_threshold','confidence_map_slope'],
							  method='pearson',nanhandling='pairwise',
							  correct_multiple_comparison_pvalues=True,
							  normalize={'internal_var':'experiment'},
							  normalization_function=lambda x: x/np.nanstd(x))
		
		Compute the correlation matrix between parameters from the
		parameter values for each experiment, session and subject name.
		
		Input:
			used_statistics: A list with the subject summary statistics
				that will be correlated with the used_parameters.
			used_parameters: A list with the parameters that will be
				used to compute the correlations.
			method: The method used to compute the correlation matrix.
				Refer to utils.corrcoef for more information.
			nanhandling: The nan handling policy used to compute the
				correlation matrix. Refer to utils.corrcoef for more
				information.
			correct_multiple_comparison_pvalues: A bool. If True, the
				pvalues are corrected for multiple comparisons using the
				Holm-Bonferroni method. If it is False, no correction is
				applied.
			normalize: None or a dict that indicates if and how the
				parameter values should be normalized. Refer to
				self.get_summary_stats_array for more information.
			normalization_function: A callable that is used to normalize
				the parameter values. Refer to
				self.get_summary_stats_array for more information.
		
		Output:
			corrs: A 2D numpy array that holds the correlation values.
				These range from -1 to 1, where 1 indicates full positive
				correlation, -1 indicates full negative correlation and
				0 indicates no correlation. The axis=0 indices (rows)
				correspond to different subject statistics and the axis=1
				(columns) correspond to the different used parameters.
				The order in which stats and parameters appear in the
				supplied 'used_statistics' and 'used_parameters' controls
				the order in which they appear in the corrs and pvals
				arrays.
			pvals: A 2D numpy array that holds the correlation statistic's
				p-value. Typically, p-values below 0.05 are considered
				indicative of significant correlation.
		
		"""
		subj,model = self.get_summary_stats_array(normalize=normalize,normalization_function=normalization_function)
		subj_stats = np.array([subj[s] for s in used_statistics])
		parameters = np.array([model[par] for par in used_parameters])
		corrs = []
		pvals = []
		for subj_stat in subj_stats:
			corr,pval = utils.corrcoef(subj_stat,parameters,method=method,nanhandling=nanhandling)
			corrs.extend(corr[0,1:])
			pvals.extend(pval[0,1:])
		corrs = np.array(corrs)
		pvals = np.array(pvals)
		if correct_multiple_comparison_pvalues:
			pvals = utils.holm_bonferroni(pvals)
		return corrs.reshape((len(used_statistics),len(used_parameters))),pvals.reshape((len(used_statistics),len(used_parameters)))
	
def cluster_analysis(analyzer_kwargs={},merge='names',filter_nans='post', tree_mode='r',show=False,extension='svg'):
	a = Analyzer(**analyzer_kwargs)
	a.get_parameter_array_from_summary(normalize={'internal_var':'experiment',\
												  'confidence_map_slope':'all',\
												  'cost':'all',\
												  'high_confidence_threshold':'all',\
												  'dead_time':'all',\
												  'dead_time_sigma':'all',\
												  'phase_out_prob':'all'})
	
	decision_parameters=['cost','internal_var','phase_out_prob']
	tree = a.cluster(merge=merge,clustered_parameters=decision_parameters,filter_nans=filter_nans)
	if show:
		tree.copy().show(tree_style=default_tree_style(mode=tree_mode,title='Decision Hierarchy'))
	tree.render('../../figs/decision_cluster.'+extension,tree_style=default_tree_style(mode=tree_mode,title='Decision Hierarchy'), layout=default_tree_layout,\
				dpi=300, units='mm', w=150)
	
	confidence_parameters=['high_confidence_threshold','confidence_map_slope']
	tree = a.cluster(merge=merge,clustered_parameters=confidence_parameters,filter_nans=filter_nans)
	if show:
		tree.copy().show(tree_style=default_tree_style(mode=tree_mode,title='Confidence Hierarchy'))
	tree.render('../../figs/confidence_cluster.'+extension,tree_style=default_tree_style(mode=tree_mode,title='Confidence Hierarchy'), layout=default_tree_layout,\
				dpi=300, units='mm', w=150)
	decision_parameters=['cost','internal_var','phase_out_prob']
	
	#~ tree = a.cluster(merge=merge,clustered_parameters=decision_parameters+confidence_parameters,filter_nans=filter_nans)
	#~ tree.copy().show(tree_style=default_tree_style(mode=tree_mode,title='D+C Hierarchy'))
	#~ tree = a.cluster(merge=merge,clustered_parameters=decision_parameters+confidence_parameters+['dead_time','dead_time_sigma'],filter_nans=filter_nans)
	#~ tree.copy().show(tree_style=default_tree_style(mode=tree_mode,title='All par Hierarchy'))
	
	show=False
	if show:
		a.controlled_scatter(scattered_parameters=decision_parameters,merge=merge)
		plt.legend(loc='best', fancybox=True, framealpha=0.5)
		a.controlled_scatter(scattered_parameters=confidence_parameters,merge=merge)
		plt.legend(loc='best', fancybox=True, framealpha=0.5)
		plt.show(True)

def parameter_correlation(analyzer_kwargs={}):
	a = Analyzer(**analyzer_kwargs)
	parameters,parameter_names,names,sessions,experiments = \
		a.get_parameter_array_from_summary(normalize={'internal_var':'experiment'})
	#~ parameters,parameter_names,names,sessions,experiments = \
		#~ a.get_parameter_array_from_summary(normalize={'internal_var':'experiment',\
													  #~ 'confidence_map_slope':'all',\
													  #~ 'cost':'all',\
													  #~ 'high_confidence_threshold':'all',\
													  #~ 'dead_time':'all',\
													  #~ 'dead_time_sigma':'all',\
													  #~ 'phase_out_prob':'all'})
	
	dtype = [(str('parameters'),'O'),(str('name'),'i'),(str('session'),'i'),(str('experiment'),experiments.dtype)]
	sort_array = [(p,int(n),int(s),e) for p,n,s,e in zip(parameters,names,sessions,experiments)]
	sort_array = np.array(sort_array,dtype=dtype)
	sort_array.sort(order=['experiment', 'session','name'])
	parameters = np.array([p.astype(np.float) for p in sort_array['parameters']]).reshape(8,-1,7)
	names = sort_array['name'].reshape(8,-1)
	sessions = sort_array['session'].reshape(8,-1)
	experiments = sort_array['experiment'].reshape(8,-1)
	
	gs1 = gridspec.GridSpec(2, 3, left=0.05, right=0.85)
	gs2 = gridspec.GridSpec(2, 1, left=0.88, right=0.90)
	c1s = []
	c2s = []
	used_parameter_names_dict = {\
			#~ 'all': ['cost','internal_var','phase_out_prob','high_confidence_threshold','confidence_map_slope','dead_time','dead_time_sigma'],\
			'all': ['cost','internal_var','phase_out_prob','high_confidence_threshold','confidence_map_slope'],\
			'decision': ['cost','internal_var','phase_out_prob'],\
			'confidence': ['high_confidence_threshold','confidence_map_slope']}
	for i,used_parameters in enumerate(['all','decision','confidence']):
		used_parameter_names = used_parameter_names_dict[used_parameters]
		parameter_inds = np.array([parameter_names.index(p) for p in used_parameter_names])
		temp = parameters[:,:,parameter_inds]
		
		# Test pearson correlation treating each subject's parameter as an independent observation
		p1 = temp.reshape((8,-1))
		c1,pval1 = utils.corrcoef(p1,method='pearson')
		#~ np.fill_diagonal(c1, np.nan)
		#~ np.fill_diagonal(pval1, np.nan)
		pval1 = correct_rho_pval(pval1)
		c1[pval1>0.05] = np.nan
		c1s.append(c1)
		
		# Test pearson correlation treating the parameters as categories
		p2 = temp.reshape((-1,temp.shape[-1])).T
		c2,pval2 = utils.corrcoef(p2,method='pearson')
		#~ np.fill_diagonal(c2, np.nan)
		#~ np.fill_diagonal(pval2, np.nan)
		pval2 = correct_rho_pval(pval2)
		c2[pval2>0.05] = np.nan
		c2s.append(c2)
		
	exp_alias = {'2AFC':'Con','Auditivo':'Aud','Luminancia':'Lum'}
	par_alias = {'cost':r'$c$',\
				'internal_var':r'$\sigma^{2}$',\
				'phase_out_prob':r'$p_{po}$',\
				'high_confidence_threshold':r'$C_{H}$',\
				'confidence_map_slope':r'$\alpha$',\
				'dead_time':r'$\tau_{c}$',\
				'dead_time_sigma':r'$\sigma_{c}$'}
	
	vmin1 = np.nanmin(np.array([np.nanmin(c) for c in c1s]))
	vmin2 = np.nanmin(np.array([np.nanmin(c) for c in c2s]))
	vmax1 = np.nanmax(np.array([np.nanmax(c) for c in c1s]))
	vmax2 = np.nanmax(np.array([np.nanmax(c) for c in c2s]))
	plt.figure(figsize=(14,10))
	for i,used_parameters in enumerate(['all','decision','confidence']):
		used_parameter_names = used_parameter_names_dict[used_parameters]
		
		c1 = c1s[i]
		c2 = c2s[i]
		ax = plt.subplot(gs1[i])
		#~ ax = plt.subplot(2,3,i+1)
		plt.imshow(c1,aspect='auto',cmap='jet',interpolation='none',extent=[0,len(c1),0,len(c1)],vmin=vmin1,vmax=vmax1)
		plt.xticks(np.arange(len(c1))+0.5,[exp_alias[str(e)]+' '+str(s) for e,s in zip(experiments[:,0],sessions[:,0])],rotation=60)
		plt.yticks(np.arange(len(c1))+0.5,[exp_alias[str(e)]+' '+str(s) for e,s in zip(experiments[:,0],sessions[:,0])][::-1])
		plt.title('Pars = '+used_parameters)
		if i==0:
			plt.colorbar(cax=plt.subplot(gs2[0]))
			plt.ylabel('Task correlation')
		
		ax = plt.subplot(gs1[i+3])
		#~ ax = plt.subplot(2,3,i+4)
		plt.imshow(c2,aspect='auto',cmap='jet',interpolation='none',extent=[0,len(c2),0,len(c2)],vmin=vmin2,vmax=vmax2)
		plt.xticks(np.arange(len(c2))+0.5,[par_alias[p] for p in used_parameter_names],rotation=60,fontsize=14)
		plt.yticks(np.arange(len(c2))+0.5,[par_alias[p] for p in used_parameter_names][::-1],fontsize=14)
		if i==0:
			plt.colorbar(cax=plt.subplot(gs2[1]))
			plt.ylabel('Parameter correlation')
	plt.show(True)

def correct_rho_pval(pvals):
	out = np.empty_like(pvals)
	out[:,:] = pvals[:,:]
	ps = []
	for rowind,row in enumerate(pvals):
		for pval in row[rowind+1:]:
			ps.append(pval)
	ps = utils.holm_bonferroni(np.array(ps))
	counter = 0
	for j,pj in enumerate(pvals):
		for k,pk in enumerate(pvals[j+1:]):
			out[j,j+k+1] = ps[counter]
			out[j+k+1,j] = ps[counter]
			counter+=1
	return out

def binary_confidence_analysis(analyzer_kwargs={}):
	a = Analyzer(**analyzer_kwargs)
	parameters,parameter_names,names,sessions,experiments = \
		a.get_parameter_array_from_summary(normalize={'internal_var':'experiment'})
	#~ parameters,parameter_names,names,sessions,experiments = \
		#~ a.get_parameter_array_from_summary(normalize={'internal_var':'experiment',\
													  #~ 'confidence_map_slope':'all',\
													  #~ 'cost':'all',\
													  #~ 'high_confidence_threshold':'all',\
													  #~ 'dead_time':'all',\
													  #~ 'dead_time_sigma':'all',\
													  #~ 'phase_out_prob':'all'})
	
	subj_rt = []
	subj_hit_rt = []
	subj_miss_rt = []
	subj_perf = []
	subj_conf = []
	subj_hit_conf = []
	subj_miss_conf = []
	subj_median_conf = []
	model_rt = []
	model_hit_rt = []
	model_miss_rt = []
	model_perf = []
	model_conf = []
	model_hit_conf = []
	model_miss_conf = []
	model_median_conf = []
	experiments = []
	sessions = []
	names = []
	summary = a.summary
	subjects = io.filter_subjects_list(io.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment')
	for k in summary['theoretical'].keys():
		vals = summary['theoretical'][k]
		subj_key = '_'.join(['experiment_'+vals['experiment'],'subject_'+str(vals['name']),'session_'+str(vals['session'])])
		subj_vals = summary['experimental'][subj_key]
		try:
			names.append(int(vals['name']))
		except:
			names.append(vals['name'])
		try:
			sessions.append(int(vals['session']))
		except:
			sessions.append(vals['session'])
		experiments.append(vals['experiment'])
		subj_rt.append(np.array([subj_vals['means']['rt'],subj_vals['stds']['rt']]))
		subj_hit_rt.append(np.array([subj_vals['means']['hit_rt'],subj_vals['stds']['hit_rt']]))
		subj_miss_rt.append(np.array([subj_vals['means']['miss_rt'],subj_vals['stds']['miss_rt']]))
		subj_perf.append(np.array([subj_vals['means']['performance'],subj_vals['stds']['performance']]))
		subj_conf.append(np.array([subj_vals['means']['confidence'],subj_vals['stds']['confidence']]))
		subj_hit_conf.append(np.array([subj_vals['means']['hit_confidence'],subj_vals['stds']['hit_confidence']]))
		subj_miss_conf.append(np.array([subj_vals['means']['miss_confidence'],subj_vals['stds']['miss_confidence']]))
		model_rt.append(vals['rt'])
		model_hit_rt.append(vals['hit_rt'])
		model_miss_rt.append(vals['miss_rt'])
		model_perf.append(vals['performance'])
		model_conf.append(vals['confidence'])
		model_hit_conf.append(vals['hit_confidence'])
		model_miss_conf.append(vals['miss_confidence'])
		
		subject = [s for s in subjects if s.experiment==vals['experiment'] and s.get_name()==str(vals['name']) and s.get_session()==str(vals['session'])][0]
		data = subject.load_data()
		subj_median_conf.append(np.median(data[:,3]))
		model_median_conf.append(vals['parameters']['high_confidence_threshold'])
	subj_rt = np.array(subj_rt)
	subj_hit_rt = np.array(subj_hit_rt)
	subj_miss_rt = np.array(subj_miss_rt)
	subj_conf = np.array(subj_conf)
	subj_hit_conf = np.array(subj_hit_conf)
	subj_miss_conf = np.array(subj_miss_conf)
	subj_perf = np.array(subj_perf)
	plt.figure(figsize=(14,10))
	plt.subplot(241)
	plt.errorbar(model_rt,subj_rt[:,0],subj_rt[:,1],marker='.',linestyle='')
	xlim = list(plt.gca().get_xlim())
	ylim = list(plt.gca().get_ylim())
	xlim[0] = min([xlim[0],ylim[0]])
	xlim[1] = max([xlim[1],ylim[1]])
	plt.plot(xlim,xlim,'--r')
	plt.gca().set_xlim(xlim)
	plt.gca().set_ylim(xlim)
	plt.ylabel('Subject')
	plt.title('Mean RT')
	
	plt.subplot(242)
	plt.errorbar(model_hit_rt,subj_hit_rt[:,0],subj_hit_rt[:,1],marker='.',linestyle='')
	xlim = list(plt.gca().get_xlim())
	ylim = plt.gca().get_ylim()
	xlim[0] = min([xlim[0],ylim[0]])
	xlim[1] = max([xlim[1],ylim[1]])
	plt.plot(xlim,xlim,'--r')
	plt.gca().set_xlim(xlim)
	plt.gca().set_ylim(xlim)
	plt.title('Mean hit RT')
	
	plt.subplot(243)
	plt.errorbar(model_miss_rt,subj_miss_rt[:,0],subj_miss_rt[:,1],marker='.',linestyle='')
	xlim = list(plt.gca().get_xlim())
	ylim = plt.gca().get_ylim()
	xlim[0] = min([xlim[0],ylim[0]])
	xlim[1] = max([xlim[1],ylim[1]])
	plt.plot(xlim,xlim,'--r')
	plt.gca().set_xlim(xlim)
	plt.gca().set_ylim(xlim)
	plt.title('Mean miss RT')
	
	plt.subplot(244)
	plt.errorbar(model_perf,subj_perf[:,0],subj_perf[:,1],marker='.',linestyle='')
	xlim = list(plt.gca().get_xlim())
	ylim = plt.gca().get_ylim()
	xlim[0] = min([xlim[0],ylim[0]])
	xlim[1] = max([xlim[1],ylim[1]])
	plt.plot(xlim,xlim,'--r')
	plt.gca().set_xlim(xlim)
	plt.gca().set_ylim(xlim)
	plt.title('Mean performance')
	
	plt.subplot(234)
	plt.errorbar(model_conf,subj_conf[:,0],subj_conf[:,1],marker='.',linestyle='')
	xlim = list(plt.gca().get_xlim())
	ylim = plt.gca().get_ylim()
	xlim[0] = min([xlim[0],ylim[0]])
	xlim[1] = max([xlim[1],ylim[1]])
	plt.plot(xlim,xlim,'--r')
	plt.gca().set_xlim(xlim)
	plt.gca().set_ylim(xlim)
	plt.ylabel('Subject')
	plt.xlabel('Model')
	plt.title('Mean Confidence')
	
	plt.subplot(235)
	plt.errorbar(model_hit_conf,subj_hit_conf[:,0],subj_hit_conf[:,1],marker='.',linestyle='')
	xlim = list(plt.gca().get_xlim())
	ylim = plt.gca().get_ylim()
	xlim[0] = min([xlim[0],ylim[0]])
	xlim[1] = max([xlim[1],ylim[1]])
	plt.plot(xlim,xlim,'--r')
	plt.gca().set_xlim(xlim)
	plt.gca().set_ylim(xlim)
	plt.title('Mean hit Confidence')
	plt.xlabel('Model')
	
	plt.subplot(236)
	plt.errorbar(model_miss_conf,subj_miss_conf[:,0],subj_miss_conf[:,1],marker='.',linestyle='')
	xlim = list(plt.gca().get_xlim())
	ylim = plt.gca().get_ylim()
	xlim[0] = min([xlim[0],ylim[0]])
	xlim[1] = max([xlim[1],ylim[1]])
	plt.plot(xlim,xlim,'--r')
	plt.gca().set_xlim(xlim)
	plt.gca().set_ylim(xlim)
	plt.title('Mean miss Confidence')
	plt.xlabel('Model')
	
	plt.show(True)

def correlation_analysis(analyzer_kwargs={}):
	a = Analyzer(**analyzer_kwargs)
	subj,model = a.get_summary_stats_array(normalize={'internal_var':'experiment'})
	#~ subj,model = a.get_summary_stats_array(normalize={'internal_var':'experiment',\
													  #~ 'confidence_map_slope':'all',\
													  #~ 'cost':'all',\
													  #~ 'high_confidence_threshold':'all',\
													  #~ 'dead_time':'all',\
													  #~ 'dead_time_sigma':'all',\
													  #~ 'phase_out_prob':'all'})
	
	ue,c = np.unique(model['experiment'],return_inverse=True)
	
	parameter_names = ['cost','internal_var','phase_out_prob','dead_time','dead_time_sigma','high_confidence_threshold','confidence_map_slope']
	parameter_aliases = {'cost':r'$c$',
						'internal_var':r'$\sigma^{2}$',
						'phase_out_prob':r'$p_{po}$',
						'high_confidence_threshold':r'$C_{H}$',
						'confidence_map_slope':r'$\alpha$',
						'dead_time':r'$\tau_{c}$',
						'dead_time_sigma':r'$\sigma_{c}$'}
	# Correlation between parameters and mean confidence
	plt.figure()
	axs = {'cost':plt.subplot(231),
			'internal_var':plt.subplot(232),
			'phase_out_prob':plt.subplot(233),
			'high_confidence_threshold':plt.subplot(245),
			'confidence_map_slope':plt.subplot(246),
			'dead_time':plt.subplot(247),
			'dead_time_sigma':plt.subplot(248)}
	ylabels = {'cost':'Mean confidence',
			'internal_var':'',
			'phase_out_prob':'',
			'high_confidence_threshold':'Mean confidence',
			'confidence_map_slope':'',
			'dead_time':'',
			'dead_time_sigma':''}
	for par in parameter_names:
		ax = axs[par]
		ax.scatter(model[par],subj['confidence_mean'],c=c)
		ax.set_xlabel(parameter_aliases[par])
		ax.set_ylabel(ylabels[par])
	plt.suptitle('Correlation between parameters and mean confidence')
	
	# Correlation between parameters and mean RT
	plt.figure()
	axs = {'cost':plt.subplot(231),
			'internal_var':plt.subplot(232),
			'phase_out_prob':plt.subplot(233),
			'high_confidence_threshold':plt.subplot(245),
			'confidence_map_slope':plt.subplot(246),
			'dead_time':plt.subplot(247),
			'dead_time_sigma':plt.subplot(248)}
	ylabels['cost'] = ylabels['high_confidence_threshold'] = 'Mean RT'
	for par in parameter_names:
		ax = axs[par]
		ax.scatter(model[par],subj['rt_mean'],c=c)
		ax.set_xlabel(parameter_aliases[par])
		ax.set_ylabel(ylabels[par])
	plt.suptitle('Correlation between parameters and mean RT')
	
	# Correlation between parameters and mean performance
	plt.figure()
	axs = {'cost':plt.subplot(231),
			'internal_var':plt.subplot(232),
			'phase_out_prob':plt.subplot(233),
			'high_confidence_threshold':plt.subplot(245),
			'confidence_map_slope':plt.subplot(246),
			'dead_time':plt.subplot(247),
			'dead_time_sigma':plt.subplot(248)}
	ylabels['cost'] = ylabels['high_confidence_threshold'] = 'Mean performance'
	for par in parameter_names:
		ax = axs[par]
		ax.scatter(model[par],subj['performance_mean'],c=c)
		ax.set_xlabel(parameter_aliases[par])
		ax.set_ylabel(ylabels[par])
	plt.suptitle('Correlation between parameters and mean performance')
	
	from scipy.stats import linregress, ttest_rel, ttest_ind, ttest_1samp
	# Agreement between data means and fit
	f = plt.figure()
	axs = {'rt':plt.subplot(231),'confidence':plt.subplot(232),
		   'performance':plt.subplot(233),'hit_rt':plt.subplot(245),
		   'miss_rt':plt.subplot(246),'hit_confidence':plt.subplot(247),
		   'miss_confidence':plt.subplot(248)}
	titles = {'rt':'Mean RT','confidence':'Mean confidence',
			  'performance':'Mean performance','hit_rt':'Mean RT for hits',
			  'miss_rt':'Mean RT for misses','hit_confidence':'Mean conf for hits',
			  'miss_confidence':'Mean conf for misses'}
	ylabels = {'rt':'Subject data','confidence':'','performance':'',
			   'hit_rt':'Subject data','miss_rt':'',
			   'hit_confidence':'','miss_confidence':''}
	xlabels = {'rt':'','confidence':'','performance':'',
			   'hit_rt':'Model data','miss_rt':'Model data',
			   'hit_confidence':'Model data','miss_confidence':'Model data'}
	for key in axs.keys():
		ax = axs[key]
		for i,exp in enumerate(ue):
			color = plt.get_cmap('jet')(float(i)/float(len(ue)-1))
			inds = c==i
			label = {'2AFC':'Contrast','Auditivo':'Auditory','Luminancia':'Luminance'}[str(exp).strip('\x00')]
			ax.errorbar(model[key+'_mean'][inds],subj[key+'_mean'][inds],subj[key+'_std'][inds],linestyle='',marker='o',color=color,label=label)
		slope,intercept,rvalue,_,stder = linregress(model[key+'_mean'],subj[key+'_mean'])
		par,cov = utils.linear_least_squares(model[key+'_mean'],subj[key+'_mean'],subj[key+'_std'])
		tvalue,pvalue = ttest_1samp(subj[key+'_mean']-model[key+'_mean'],0.)
		title = titles[key]
		#~ if pvalue<0.05:
			#~ title+=' *'
			#~ if pvalue<0.005:
				#~ title+='*'
				#~ if pvalue<0.0005:
					#~ title+='*'
		ax.set_title(title)
		ax.set_xlabel(xlabels[key])
		ax.set_ylabel(ylabels[key])
		xlim = np.array(ax.get_xlim())
		ylim = np.array(ax.get_ylim())
		xlim[0] = min([xlim[0],ylim[0]])
		xlim[1] = max([xlim[1],ylim[1]])
		ylim = xlim
		xlsq = np.linspace(xlim[0],xlim[1],1000)
		ylsq,sylsq = utils.linear_least_squares_prediction(xlsq,par,cov)
		ax.plot(xlsq,ylsq,color='gray',linewidth=2)
		ax.fill_between(xlsq,ylsq-2*sylsq,ylsq+2*sylsq,facecolor='gray',alpha=0.4,interpolate=True)
		ax.plot(xlim,xlim,color='k',linewidth=2)
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		if key=='rt':
			ax.legend(loc='best', fancybox=True, framealpha=0.5)
	plt.figure(f.number)
	utils.maximize_figure()
	plt.suptitle('Agreement between data means and fit',fontsize=18)
	
	# Agreement between data medians and fit
	plt.figure()
	axs = {'rt':plt.subplot(221),'confidence':plt.subplot(222),
		   'hit_rt':plt.subplot(245),'miss_rt':plt.subplot(246),
		   'hit_confidence':plt.subplot(247),'miss_confidence':plt.subplot(248)}
	titles = {'rt':'Median RT','confidence':'Median confidence',
			  'hit_rt':'Median RT for hits','miss_rt':'Median RT for misses',
			  'hit_confidence':'Median conf for hits',
			  'miss_confidence':'Median conf for misses'}
	ylabels = {'rt':'Subject data','confidence':'',
			   'hit_rt':'Subject data','miss_rt':'',
			   'hit_confidence':'','miss_confidence':''}
	xlabels = {'rt':'','confidence':'',
			   'hit_rt':'Model data','miss_rt':'Model data',
			   'hit_confidence':'Model data','miss_confidence':'Model data'}
	for key in axs.keys():
		ax = axs[key]
		for i,exp in enumerate(ue):
			color = plt.get_cmap('jet')(float(i)/float(len(ue)-1))
			inds = c==i
			label = {'2AFC':'Contrast','Auditivo':'Auditory','Luminancia':'Luminance'}[str(exp).strip('\x00')]
			ax.errorbar(model[key+'_median'][inds],subj[key+'_median'][inds],subj[key+'_std'][inds],linestyle='',marker='o',color=color,label=label)
		#~ slope,intercept,rvalue,_,stder = linregress(model[key+'_median'],subj[key+'_median'])
		par,cov = utils.linear_least_squares(model[key+'_median'],subj[key+'_median'],subj[key+'_std'])
		tvalue,pvalue = ttest_1samp(subj[key+'_median']-model[key+'_median'],0.)
		#~ print(key,np.mean(subj[key+'_median']-model[key+'_median']),tvalue,pvalue)
		title = titles[key]
		if key=='rt':
			ax.legend(loc='best', fancybox=True, framealpha=0.5)
		#~ if pvalue<0.05:
			#~ title+=' *'
			#~ if pvalue<0.005:
				#~ title+='*'
				#~ if pvalue<0.0005:
					#~ title+='*'
		ax.set_title(title)
		ax.set_xlabel(xlabels[key])
		ax.set_ylabel(ylabels[key])
		xlim = np.array(ax.get_xlim())
		ylim = np.array(ax.get_ylim())
		xlim[0] = min([xlim[0],ylim[0]])
		xlim[1] = max([xlim[1],ylim[1]])
		ylim = xlim
		xlsq = np.linspace(xlim[0],xlim[1],1000)
		ylsq,sylsq = utils.linear_least_squares_prediction(xlsq,par,cov)
		ax.plot(xlsq,ylsq,color='gray',linewidth=2)
		ax.fill_between(xlsq,ylsq-2*sylsq,ylsq+2*sylsq,facecolor='gray',alpha=0.4,interpolate=True)
		ax.plot(xlim,xlim,color='k',linewidth=2)
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
	utils.maximize_figure()
	plt.suptitle('Agreement between data medians and fit',fontsize=18)
	
	# Correlation between parameters and: mean RT, performance, mean confidence, AUC and multi mod index
	studied_parameters = [par for par in parameter_names if par not in ['dead_time','dead_time_sigma']]
	parameters = np.array([model[par] for par in studied_parameters])
	group_width = 0.9
	bar_width = group_width/5
	bar_pos = np.arange(parameters.shape[0])
	group_center = bar_pos+0.5*group_width
	plt.figure()
	colors = ['r','g','b','y','m']
	studied_stats = ['rt_mean','performance_mean','confidence_mean','auc','multi_mod_index']
	stat_aliases = {'rt_mean':'Mean RT','performance_mean':'Performance',
					'confidence_mean':'Mean Confidence',
					'auc':'AUC','multi_mod_index':"Hartigan's DIP"}
	bars = []
	pvals = []
	signs = []
	ax1 = plt.subplot(211)
	ax2 = plt.subplot(212)
	for i,stat in enumerate(studied_stats):
		corr,pval = utils.corrcoef(subj[stat],parameters)
		corr = corr[0,1:]
		pval = pval[0,1:]
		pvals.append(pval)
		bars.append(ax1.bar(bar_pos+i*bar_width,np.abs(corr),bar_width,color=colors[i]))
		ax2.bar(bar_pos+i*bar_width,corr,bar_width,color=colors[i])
		signs.append(np.sign(corr))
	pvals = np.array(pvals)
	sh = pvals.shape
	pvals = utils.holm_bonferroni(pvals.reshape((-1))).reshape(sh)
	for rects,pval,sign in zip(bars,pvals,signs):
		for rect,p,si in zip(rects,pval,sign):
			significance_mark = ''
			if p<0.05:
				significance_mark+='*'
				if p<0.005:
					significance_mark+='*'
					if p<0.0005:
						significance_mark+='*'
			height = rect.get_height()
			x = rect.get_x() + rect.get_width()/2.
			ax1.text(rect.get_x() + rect.get_width()/2., height, significance_mark,ha='center', va='bottom')
			ax2.text(rect.get_x() + rect.get_width()/2., height*si, significance_mark,ha='center', va='bottom' if si>=0 else 'top')
	ax1.set_xticks(group_center)
	ax2.set_xticks(group_center)
	ax1.set_xticklabels([parameter_aliases[par] for par in studied_parameters],fontsize=16)
	ax2.set_xticklabels([parameter_aliases[par] for par in studied_parameters],fontsize=16)
	ax1.set_ylabel('Absolute correlation')
	ax2.set_ylabel('Correlation')
	ax1.legend([r[0] for r in bars],[stat_aliases[stat] for stat in studied_stats],loc='best', fancybox=True, framealpha=0.5)
	ax2.plot(ax2.get_xlim(),[0,0],'k')
	utils.maximize_figure()
	plt.show(True)

def compare_mappings():
	experiment_alias = {'2AFC':'Con','Auditivo':'Aud','Luminancia':'Lum'}
	all_lo = Analyzer(cmap_meth='log_odds').get_summary_stats_array()[1]
	# For some reason we dont have the fit result for the linear mapping
	# of experiment Luminancia subject 12 and session 1 (index 113 of lo)
	all_lo = np.concatenate((all_lo[:113],all_lo[114:]),axis=0)
	all_li = Analyzer(cmap_meth='belief').get_summary_stats_array()[1]
	cat = 'experiment'
	ucat_lo = np.unique(all_lo[cat])
	ucat_li = np.unique(all_li[cat])
	output = [['Experiment',r'$nLL\left(\mathcal{C}_{\mathcal{L}_{o}}\right)$',r'$nLL\left(\mathcal{C}_{s}\right)$',r'$2\log\left(\frac{\mathcal{L}(\mathcal{C}_{s})}{\mathcal{L}(\mathcal{C}_{\mathcal{L}_{o}})}\right)$']]
	all_lo_nLL = 0.
	all_li_nLL = 0.
	for ucat in ucat_lo:
		lo = all_lo[ucat==all_lo[cat]]
		li = all_li[ucat==all_li[cat]]
		total_lo_nLL = np.sum(lo['full_confidence_merit'])
		total_li_nLL = np.sum(li['full_confidence_merit'])
		all_lo_nLL+= total_lo_nLL
		all_li_nLL+= total_li_nLL
		percent_of_low_li = np.sum((lo['full_confidence_merit']>li['full_confidence_merit']).astype(np.float))/float(len(lo))*100
		log_likelihood_ratio = lo['full_confidence_merit']-li['full_confidence_merit'] # The stored values are nLL so this is equal to log(li_like/lo_like)
		#~ print(ucat)
		#~ print('Overall log_odds mapping nLL = {0}'.format(total_lo_nLL))
		#~ print('Overall linear mapping nLL = {0}'.format(total_li_nLL))
		#~ print('Percent of experiment,subject,session tuples that are better explained with the linear mapping = {0}%'.format(percent_of_low_li))
		#~ print('Mean log likelihood ratio in favor of linear mapping = {0}'.format(np.mean(log_likelihood_ratio)))
		#~ print('Likelihood ratio T-value = {0}'.format(np.exp(np.mean(log_likelihood_ratio))))
		#~ print('Likelihood ratio p-value = {0}'.format(0.5*(1-stats.t.cdf(np.exp(np.mean(log_likelihood_ratio)),1))))
		#~ print('Total double log likelihood ratio = {0}'.format(2*np.sum(log_likelihood_ratio)))
		#~ print('Total log likelihood ratio Wilk´s p-value = {0}'.format(0.5*(1-stats.chi2.cdf(2*np.sum(log_likelihood_ratio),1))))
		
		#~ plt.hist(log_likelihood_ratio)
		#~ plt.xlabel('nLL(log_odds)-nLL(linear)')
		#~ plt.show(True)
		output.append([experiment_alias[str(ucat).replace(' x00','')],'{0:.2f}'.format(total_lo_nLL),'{0:.2f}'.format(total_li_nLL),'{0:.2f}'.format(2*np.sum(log_likelihood_ratio))])
	output.append([u'All','{0:.2f}'.format(all_lo_nLL),'{0:.2f}'.format(all_li_nLL),'{0:.2f}'.format(2*(all_lo_nLL-all_li_nLL))])
	output = ' \\\\ \\hline\n'.join([' & '.join(x) for x in output])+' \\\\ \\hline\n'
	print(output)
	
	
	cat = 'name'
	ucat_lo = np.unique(all_lo[cat])
	ucat_li = np.unique(all_li[cat])
	output = [['Subject id',r'$nLL\left(\mathcal{C}_{\mathcal{L}_{o}}\right)$',r'$nLL\left(\mathcal{C}_{s}\right)$',r'$2\log\left(\frac{\mathcal{L}(\mathcal{C}_{s})}{\mathcal{L}(\mathcal{C}_{\mathcal{L}_{o}})}\right)$']]
	all_lo_nLL = 0.
	all_li_nLL = 0.
	dtype = ucat_lo.dtype
	ucat_lo = np.sort(ucat_lo.astype(np.int)).astype(dtype)
	for ucat in ucat_lo:
		lo = all_lo[ucat==all_lo[cat]]
		li = all_li[ucat==all_li[cat]]
		total_lo_nLL = np.sum(lo['full_confidence_merit'])
		total_li_nLL = np.sum(li['full_confidence_merit'])
		all_lo_nLL+= total_lo_nLL
		all_li_nLL+= total_li_nLL
		percent_of_low_li = np.sum((lo['full_confidence_merit']>li['full_confidence_merit']).astype(np.float))/float(len(lo))*100
		log_likelihood_ratio = lo['full_confidence_merit']-li['full_confidence_merit'] # The stored values are nLL so this is equal to log(li_like/lo_like)
		#~ print(ucat)
		#~ print('Overall log_odds mapping nLL = {0}'.format(total_lo_nLL))
		#~ print('Overall linear mapping nLL = {0}'.format(total_li_nLL))
		#~ print('Percent of experiment,subject,session tuples that are better explained with the linear mapping = {0}%'.format(percent_of_low_li))
		#~ print('Mean log likelihood ratio in favor of linear mapping = {0}'.format(np.mean(log_likelihood_ratio)))
		#~ print('Likelihood ratio T-value = {0}'.format(np.exp(np.mean(log_likelihood_ratio))))
		#~ print('Likelihood ratio p-value = {0}'.format(0.5*(1-stats.t.cdf(np.exp(np.mean(log_likelihood_ratio)),1))))
		#~ print('Total double log likelihood ratio = {0}'.format(2*np.sum(log_likelihood_ratio)))
		#~ print('Total log likelihood ratio Wilk´s p-value = {0}'.format(0.5*(1-stats.chi2.cdf(2*np.sum(log_likelihood_ratio),1))))
		
		#~ plt.hist(log_likelihood_ratio)
		#~ plt.xlabel('nLL(log_odds)-nLL(linear)')
		#~ plt.show(True)
		output.append([str(ucat).replace(' x00',''),'{0:.2f}'.format(total_lo_nLL),'{0:.2f}'.format(total_li_nLL),'{0:.2f}'.format(2*np.sum(log_likelihood_ratio))])
	output.append([u'All','{0:.2f}'.format(all_lo_nLL),'{0:.2f}'.format(all_li_nLL),'{0:.2f}'.format(2*(all_lo_nLL-all_li_nLL))])
	output = ' \\\\ \\hline\n'.join([' & '.join(x) for x in output])+' \\\\ \\hline\n'
	print(output)
	
	
	label_ind,ucat_lo_inds = utils.unique_rows(np.hstack((all_lo['experiment'][:,None],all_lo['session'][:,None])),return_index=True,return_inverse=True)[1:]
	ucat_li_inds = utils.unique_rows(np.hstack((all_li['experiment'][:,None],all_li['session'][:,None])),return_inverse=True)[1]
	output = [['Experiment session',r'$nLL\left(\mathcal{C}_{\mathcal{L}_{o}}\right)$',r'$nLL\left(\mathcal{C}_{s}\right)$',r'$2\log\left(\frac{\mathcal{L}(\mathcal{C}_{s})}{\mathcal{L}(\mathcal{C}_{\mathcal{L}_{o}})}\right)$']]
	all_lo_nLL = 0.
	all_li_nLL = 0.
	ucat_label = sorted([' '.join([experiment_alias[str(x['experiment']).strip(' \x00')],'Ses='+str(x['session']).strip(' \x00')]) for x in all_lo[label_ind]])
	dtype = ucat_lo.dtype
	ucat_lo = np.sort(ucat_lo.astype(np.int)).astype(dtype)
	for ind,ucat in enumerate(ucat_label):
		lo = all_lo[ucat_lo_inds==ind]
		li = all_li[ucat_li_inds==ind]
		total_lo_nLL = np.sum(lo['full_confidence_merit'])
		total_li_nLL = np.sum(li['full_confidence_merit'])
		all_lo_nLL+= total_lo_nLL
		all_li_nLL+= total_li_nLL
		percent_of_low_li = np.sum((lo['full_confidence_merit']>li['full_confidence_merit']).astype(np.float))/float(len(lo))*100
		log_likelihood_ratio = lo['full_confidence_merit']-li['full_confidence_merit'] # The stored values are nLL so this is equal to log(li_like/lo_like)
		#~ print(ucat)
		#~ print('Overall log_odds mapping nLL = {0}'.format(total_lo_nLL))
		#~ print('Overall linear mapping nLL = {0}'.format(total_li_nLL))
		#~ print('Percent of experiment,subject,session tuples that are better explained with the linear mapping = {0}%'.format(percent_of_low_li))
		#~ print('Mean log likelihood ratio in favor of linear mapping = {0}'.format(np.mean(log_likelihood_ratio)))
		#~ print('Likelihood ratio T-value = {0}'.format(np.exp(np.mean(log_likelihood_ratio))))
		#~ print('Likelihood ratio p-value = {0}'.format(0.5*(1-stats.t.cdf(np.exp(np.mean(log_likelihood_ratio)),1))))
		#~ print('Total double log likelihood ratio = {0}'.format(2*np.sum(log_likelihood_ratio)))
		#~ print('Total log likelihood ratio Wilk´s p-value = {0}'.format(0.5*(1-stats.chi2.cdf(2*np.sum(log_likelihood_ratio),1))))
		
		#~ plt.hist(log_likelihood_ratio)
		#~ plt.xlabel('nLL(log_odds)-nLL(linear)')
		#~ plt.show(True)
		output.append([str(ucat).replace(' x00',''),'{0:.2f}'.format(total_lo_nLL),'{0:.2f}'.format(total_li_nLL),'{0:.2f}'.format(2*np.sum(log_likelihood_ratio))])
	output.append([u'All','{0:.2f}'.format(all_lo_nLL),'{0:.2f}'.format(all_li_nLL),'{0:.2f}'.format(2*(all_lo_nLL-all_li_nLL))])
	output = ' \\\\ \\hline\n'.join([' & '.join(x) for x in output])+' \\\\ \\hline\n'
	print(output)

def mapping_strengths_and_weaknesses(analyzer_kwargs={}):
	belief_kwargs = analyzer_kwargs.copy()
	belief_kwargs['cmap_meth'] = 'belief'
	log_odds_kwargs = analyzer_kwargs.copy()
	log_odds_kwargs['cmap_meth'] = 'log_odds'
	ab = Analyzer(**belief_kwargs)
	alo = Analyzer(**log_odds_kwargs)
	teob,expb = ab.get_summary_stats_array()
	teolo,explo = alo.get_summary_stats_array()
	
	compared_keys = ['rt_mean', 'confidence_mean', 'auc',
					 'hit_confidence_mean', 'miss_confidence_mean',
					 'hit_rt_mean', 'miss_rt_mean']
	plt.figure()
	for ind,k in enumerate(compared_keys):
		ax = plt.subplot(2,4,ind)
		if k.endswith('_mean'):
			ax.errorbar(teob[k],expb[k],expb[k.replace('_mean','_std')],'.b')
			ax.errorbar(teolo[k],explo[k],explo[k.replace('_mean','_std')],'.r')
		else:
			ax.plot(teob[k],expb[k],'ob')
			ax.plot(teolo[k],explo[k],'or')
		lims = [0,0]
		lims[0] = np.min([ax.get_xlim()[0],ax.get_ylim()[0]])
		lims[1] = np.max([ax.get_xlim()[1],ax.get_ylim()[1]])
		ax.plot(lims,lims,'--k')
		ax.set_xlim(lims)
		ax.set_ylim(lims)
	plt.show(True)

def test():
	mapping_strengths_and_weaknesses()
	#~ compare_mappings()
	return
	a = Analyzer(cmap_meth='belief')
	a.get_parameter_array_from_summary(normalize={'internal_var':'experiment','dead_time':'name','dead_time_sigma':'session'})
	tree = a.cluster(merge='names',clustered_parameters=['high_confidence_threshold','confidence_map_slope'])
	#~ tree.copy().render('cluster_test.svg',tree_style=default_tree_style(mode='r'), layout=default_tree_layout)
	tree.show(tree_style=default_tree_style(mode='r'))
	tree = a.cluster(merge='sessions',clustered_parameters=['high_confidence_threshold','confidence_map_slope'])
	#~ tree.copy().render('cluster_test.svg',tree_style=default_tree_style(mode='c'), layout=default_tree_layout)
	tree.show(tree_style=default_tree_style(mode='c'))
	tree = a.cluster(clustered_parameters=['high_confidence_threshold','confidence_map_slope'])
	#~ tree.copy().render('cluster_test.svg',tree_style=default_tree_style(mode='c'), layout=default_tree_layout)
	tree.show(tree_style=default_tree_style(mode='c'))
	a.scatter_parameters(show=True)
	a.scatter_parameters(merge='names',show=True)
	a.scatter_parameters(merge='sessions',show=True)
	a.set_pooling_func(np.median)
	a.scatter_parameters(show=True)
	a.scatter_parameters(merge='names',show=True)
	a.scatter_parameters(merge='sessions',show=True)
	#~ unams,indnams = np.unique(a._names,return_inverse=True)
	#~ uexps,indexps = np.unique(a._experiments,return_inverse=True)
	#~ usess,indsess = np.unique(a._sessions,return_inverse=True)
	
	#~ for un in unams:
		#~ inds = a._names==un
		#~ pars = a._parameters[inds]
		#~ cost = pars[:,a._parameter_names.index('cost')]
		#~ internal_var = pars[:,a._parameter_names.index('internal_var')]
		#~ phase_out_prob = pars[:,a._parameter_names.index('phase_out_prob')]
		#~ high_confidence_threshold = pars[:,a._parameter_names.index('high_confidence_threshold')]
		#~ confidence_map_slope = pars[:,a._parameter_names.index('confidence_map_slope')]
		#~ plt.figure(figsize=(10,10))
		#~ ax1 = plt.subplot(221)
		#~ ax2 = plt.subplot(222)
		#~ ax3 = plt.subplot(223)
		#~ ax4 = plt.subplot(224)
		#~ for us,marker in zip(usess,['o','s','D']):
			#~ inds2 = a._sessions[inds]==us
			#~ if any(inds2):
				#~ colors = [{'2AFC':'r','Auditivo':'g','Luminancia':'b'}[str(x).strip('\x00')] for x in a._experiments[inds][inds2]]
				#~ ax1.scatter(cost[inds2],internal_var[inds2],c=colors,s=20,label=us,marker=marker)
				#~ ax2.scatter(cost[inds2],phase_out_prob[inds2],c=colors,s=20,label=us,marker=marker)
				#~ ax3.scatter(internal_var[inds2],phase_out_prob[inds2],c=colors,s=20,label=us,marker=marker)
				#~ ax4.scatter(confidence_map_slope[inds2],high_confidence_threshold[inds2],c=colors,s=20,label=us,marker=marker)
		#~ ax1.set_xlabel('cost')
		#~ ax1.set_ylabel('internal_var')
		#~ ax2.set_xlabel('cost')
		#~ ax2.set_ylabel('phase_out_prob')
		#~ ax3.set_xlabel('internal_var')
		#~ ax3.set_ylabel('phase_out_prob')
		#~ ax3.legend()
		#~ ax4.set_xlabel('confidence_map_slope')
		#~ ax4.set_ylabel('high_confidence_threshold')
		#~ plt.suptitle('Subject: '+str(un))
		#~ plt.show(True)

def parse_input():
	script_help = """ moving_bounds_fits.py help
 Sintax:
 moving_bounds_fits.py [option flag] [option value]
 
 moving_bounds_fits.py -h [or --help] displays help
 
 Optional arguments are:
 '--show': This flag takes no values. If present it displays the plotted figure
           and freezes execution until the figure is closed.
 '--test': This flag takes no values. If present the script's testsuite is
           executed.
 '-w': Override the existing saved summary file. If the flag '-w' is
       supplied, the script  will override the saved summary file. If this
       flag is not supplied, the script will attempt to load the summary
       and if it fails, it will produce the summary file. WARNING:
       the generation of the summary file takes a very long time.
 '-m' or '--method': String that identifies the fit method. Available values are full,
                     confidence_only and full_confidence. [Default 'full_confidence']
 '-o' or '--optimizer': String that identifies the optimizer used for fitting.
                        Available values are 'cma' and all the scipy.optimize.minimize methods.
                        [Default 'cma']
 '-sf' or '--suffix': A string suffix to paste to the filenames. [Default '']
 '-e' or '--extension': A string that determines the graphics fileformat
                        in which the tree graph will be saved. Available
                        extensions are 'pdf', 'png' or 'svg'. [Default 'svg']
 '-n' or '--n_clusters': An integer that specifies the number of clusters
                         constructed by the scikit-learn AgglomerativeClustering
                         class. [Default 2]
 '-a' or '--affinity': The scikit-learn AgglomerativeClustering class affinity
                       posible values are 'euclidean', 'l1', 'l2', 'cosine',
                       'manhattan' or 'precomputed'. If linkage is 'ward',
                       only 'euclidean' is accepted. Refer to the scikit-learn
                       documentation for more information. [Default 'euclidean']
 '-l' or '--linkage': The scikit-learn AgglomerativeClustering class linkage
                      posible values are 'ward', 'complete' or 'average'.
                      Refer to the scikit-learn documentation for more
                      information. [Default 'ward']
 '-pf' or '--pooling_func': The scikit-learn AgglomerativeClustering class pooling_func.
                            Default is np.nanmean (notice that numpy is
                            aliased as np when supplying an option).
                            The pooling_func is also used when scattering
                            the parameters but this functionality is only
                            accesible when importing the analysis.py package).
 '--merge': Can be 'none', 'experiments', 'sessions' or 'names'. This option
            controls if the fitted model parameters should be pooled together
            or not, and how they should be pooled. If 'None', the parameters are not
            pooled together. The parameters have three separate categories:
            the experiment to which they belong, the subject name and the
            experimental session. If the supplied option value is 'experiments',
            'sessions' or 'names', the parameters that belong to different
            categories of the supplied option value will be pooled together
            using the pooling_func. For example, if the option value is
            'names', the parameters for will still distinguish the experiment
            and session, but the parameters for different subject names will
            be pooled together. [Default 'names']
 '-f' or '--filter_nans': Can be 'pre', 'post' or 'none'. This option controls
                          how to filter the parameters that are nans. If 'none',
                          no filter is applied. If 'pre', the parameters
                          that contain a nan entry are filtered before
                          reducing the parameters to the clustered parameter
                          space. If 'post', the parameters that contain a
                          nan entry are filtered after the reduction takes
                          place. [Default 'post']
 '-t' or '--tree_mode': Can be 'r' or 'c' and controls how to plot the
                         cluster hierarchy. If 'r', the tree is plotted
                         in rectangular mode. If 'c', the tree is plotted
                         in circular mode.
 Example:
 python analysis.py --show -pf np.nanmedian"""
	str_caster = lambda x: str(x).lower()
	int_caster = int
	evaler = eval
	available_options_casters = {'method':str_caster,\
								'optimizer':str_caster,\
								'suffix':str_caster,\
								'extension':str_caster,\
								'override':None,\
								'n_clusters':int_caster,\
								'affinity':str_caster,\
								'linkage':str_caster,\
								'pooling_func':evaler,\
								'merge':str_caster,\
								'filter_nans':str_caster,\
								'tree_mode':str_caster,\
								'show':None,\
								'test':None}
	options =  {'test':False,'override':False,'show':False}
	expecting_key = True
	key = None
	if len(sys.argv)==1:
		options['test'] = True
	for i,arg in enumerate(sys.argv[1:]):
		if expecting_key:
			if arg=='--test':
				options['test'] = True
			elif arg=='-w' or arg=='--override':
				options['override'] = True
			elif arg=='--show':
				options['show'] = True
			elif arg=='-m' or arg=='--method':
				key = 'method'
				expecting_key = False
			elif arg=='-o' or arg=='--optimizer':
				key = 'optimizer'
				expecting_key = False
			elif arg=='-sf' or arg=='--suffix':
				key = 'suffix'
				expecting_key = False
			elif arg=='-e' or arg=='--extension':
				key = 'extension'
				expecting_key = False
			elif arg=='-n' or arg=='--n_clusters':
				key = 'n_clusters'
				expecting_key = False
			elif arg=='-a' or arg=='--affinity':
				key = 'affinity'
				expecting_key = False
			elif arg=='-l' or arg=='--linkage':
				key = 'linkage'
				expecting_key = False
			elif arg=='-pf' or arg=='--pooling_func':
				key = 'pooling_func'
				expecting_key = False
			elif arg=='--merge':
				key = 'merge'
				expecting_key = False
			elif arg=='-f' or arg=='--filter_nans':
				key = 'filter_nans'
				expecting_key = False
			elif arg=='-t' or arg=='--tree_mode':
				key = 'tree_mode'
				expecting_key = False
			elif arg=='-h' or arg=='--help':
				print(script_help)
				sys.exit()
			else:
				raise RuntimeError("Unknown option: {opt} encountered in position {pos}. Refer to the help to see the list of options".format(opt=arg,pos=i+1))
		else:
			expecting_key = True
			options[key] = available_options_casters[key](arg)
	try:
		if options['merge']=='none':
			options['merge'] = None
	except:
		pass
	if not expecting_key:
		raise RuntimeError("Expected a value after encountering key '{0}' but no value was supplied".format(arg))
	return options

if __name__=="__main__":
	# Parse input from sys.argv
	options = parse_input()
	if options['test']:
		test()
	else:
		del options['test']
		cluster_analysis(**options)
