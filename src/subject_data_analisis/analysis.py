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
import os, re, pickle, warnings, json, logging, copy, scipy.integrate, itertools, ete3
from sklearn import cluster

class Analyzer():
	def __init__(self,method = 'full_confidence', optimizer = 'cma', suffix = '', override=False,\
				n_clusters=2, affinity='euclidean', connectivity=None, linkage='ward', pooling_func=np.nanmean):
		self.method = method
		self.optimizer = optimizer
		self.suffix = suffix
		self.get_summary(override=override)
		self.init_clusterer(n_clusters=n_clusters, affinity=affinity,\
				connectivity=connectivity, linkage=linkage, pooling_func=pooling_func)
	
	def init_clusterer(self,n_clusters=2, affinity='euclidean',connectivity=None,\
						compute_full_tree=True, linkage='ward', pooling_func=np.nanmean):
		self.agg_clusterer = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity,\
				connectivity=connectivity,compute_full_tree=compute_full_tree, linkage=linkage,\
				pooling_func=pooling_func)
		self.linkage = linkage
		self.affinity = affinity
		self.pooling_func=pooling_func
	
	def set_pooling_func(self,pooling_func):
		self.agg_clusterer.set_params(**{'pooling_func':pooling_func})
		self.pooling_func = pooling_func
	
	def get_summary(self,override=False):
		if override or not(os.path.exists('summary_statistics.pkl') and os.path.isfile('summary_statistics.pkl')):
			subjects = io.filter_subjects_list(io.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment')
			self.summary = {'experimental':{},'theoretical':{}}
			for s in subjects:
				print(s.get_key())
				fname = fits.Fitter_filename(s.experiment,self.method,s.get_name(),s.get_session(),self.optimizer,self.suffix)
				if not(os.path.exists(fname) and os.path.isfile(fname)):
					continue
				s_measures = self.subjectSession_measures(s)
				self.summary['experimental'].update(s_measures)
				fitter = fits.load_Fitter_from_file(fname)
				f_measures = self.fitter_measures(fitter)
				self.summary['theoretical'].update(f_measures)
			f = open('summary_statistics.pkl','w')
			pickle.dump(self.summary,f)
			f.close()
		else:
			f = open('summary_statistics.pkl','r')
			self.summary = pickle.load(f)
			f.close()
		return self.summary
	
	def subjectSession_measures(self,subjectSession):
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
	
	def fitter_measures(self,fitter):
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
	
	def get_parameter_array_from_summary(self,summary=None,normalize={'internal_var':'experiment'},normalization_function=np.nanstd):
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
				vals.append(pd[pn])
			self._parameters.append(np.array(vals))
		self._parameters = np.array(self._parameters)
		self._names = np.array(self._names)
		self._sessions = np.array(self._sessions)
		self._experiments = np.array(self._experiments)
		if normalize:
			self.normalize_internal_vars(normalize,normalization_function)
		return self._parameters,self._parameter_names,self._names,self._sessions,self._experiments
	
	def normalize_internal_vars(self,normalize,normalization_function=np.nanstd):
		for parameter in normalize.keys():
			grouping_method = normalize[parameter]
			par_ind = self._parameter_names.index(parameter)
			if grouping_method=='all':
				categories = ['all']
				categories_inds = [np.ones(self._parameters.shape[0],dtype=np.bool)]
			elif grouping_method=='experiment':
				categories_inds = []
				categories,inverse = np.unique(self._experiments,return_inverse=True)
				for i,category in enumerate(categories):
					categories_inds.append(inverse==i)
			elif grouping_method=='session':
				categories_inds = []
				categories,inverse = np.unique(self._sessions,return_inverse=True)
				for i,category in enumerate(categories):
					categories_inds.append(inverse==i)
			elif grouping_method=='name':
				categories_inds = []
				categories,inverse = np.unique(self._names,return_inverse=True)
				for i,category in enumerate(categories):
					categories_inds.append(inverse==i)
			else:
				raise ValueError('Unknown normalization grouping method: {0}'.format(grouping_method))
			for i,category_ind in enumerate(categories_inds):
				self._parameters[category_ind,par_ind]/=normalization_function(self._parameters[category_ind,par_ind])
	
	def scatter_parameters(self,merge=None,show=False):
		try:
			unames,indnames = np.unique(self._names,return_inverse=True)
		except:
			self.get_parameter_array_from_summary()
			unames,indnames = np.unique(self._names,return_inverse=True)
		uexps,indexps = np.unique(self._experiments,return_inverse=True)
		usess,indsess = np.unique(self._sessions,return_inverse=True)
		if not merge is None:
			if merge=='subjects':
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
	
	def cluster(self,merge=None,clustered_parameters=['cost','internal_var','phase_out_prob']):
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
		X = X[:,clustered_parameters_inds]
		self.agg_clusterer.fit(X)
		newick_tree = self.build_Newick_tree(self.agg_clusterer.children_,self.agg_clusterer.n_leaves_,X,leaf_labels)
		tree = ete3.Tree(newick_tree)
		return tree
	
	def build_Newick_tree(self,children,n_leaves,X,leaf_labels):
		return self.go_down_tree(children,n_leaves,X,leaf_labels,len(children)+n_leaves-1)[0]+';'
	
	def go_down_tree(self,children,n_leaves,X,leaf_labels,nodename):
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
		if self.affinity=='euclidean':
			return np.sum((branchsamples-self.pooling_func(branchsamples,axis=0))**2)
		else:
			raise NotImplementedError('get_branch_span is not implemented for affinity: {0}'.format(self.affinity))

def default_tree_layout(node):
	if node.is_leaf():
		portions = node.name.split('_')
		experiment = None
		subject = None
		session = None
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
		if experiment:
			bgcolor = {'2AFC':'#FF0000','Auditivo':'#008000','Luminancia':'#0000FF'}[experiment]
			fgcolor = {'2AFC':'#000000','Auditivo':'#000000','Luminancia':'#FFFFFF'}[experiment]
		elif session:
			bgcolor = {'1':'#FF0000','2':'#008000','3':'#0000FF'}[session]
			fgcolor = {'1':'#000000','2':'#000000','3':'#FFFFFF'}[session]
		else:
			bgcolor = '#FFFFFF'
		node.set_style(ete3.NodeStyle(bgcolor=bgcolor))
		ete3.add_face_to_node(ete3.TextFace(node.name,fgcolor=fgcolor), node, column=0, position="branch-right")

def default_tree_style(mode='r'):
	tree_style = ete3.TreeStyle()
	tree_style.layout_fn = default_tree_layout
	tree_style.show_leaf_name = False
	tree_style.show_scale = False
	#~ tree_style.show_branch_length = True
	if mode=='r':
		tree_style.rotation = 90
		tree_style.branch_vertical_margin = 10
	else:
		tree_style.mode = 'c'
		tree_style.arc_start = 0 # 0 degrees = 3 o'clock
		tree_style.arc_span = 180
	return tree_style

def test():
	a = Analyzer()
	#~ a.get_parameter_array_from_summary(normalize={'internal_var':'experiment','confidence_map_slope':'all','dead_time':'name','dead_time_sigma':'session'})
	a.get_parameter_array_from_summary(normalize={'internal_var':'experiment'})
	
	#~ a.scatter_parameters(show=True)
	#~ a.scatter_parameters(merge='subjects',show=True)
	#~ a.scatter_parameters(merge='sessions',show=True)
	
	clustered_parameters=['high_confidence_threshold','confidence_map_slope']
	a.set_pooling_func(np.median)
	style = default_tree_style(mode='r')
	tree = a.cluster(merge='names',clustered_parameters=clustered_parameters)
	#~ tree.render('cluster_test.svg',tree_style=style, layout=default_tree_layout)
	tree.show(tree_style=style)
	#~ tree = a.cluster(merge='sessions')
	#~ tree.render('cluster_test.svg',tree_style=style, layout=default_tree_layout)
	#~ tree.show(tree_style=style)
	#~ tree = a.cluster()
	#~ tree.render('cluster_test.svg',tree_style=style, layout=default_tree_layout)
	#~ tree.show(tree_style=style)
	#~ a.scatter_parameters()
	#~ a.scatter_parameters(merge='subjects')
	#~ a.scatter_parameters(merge='sessions')
	#~ a.set_pooling_func(np.median)
	#~ print(a.cluster())
	#~ print(a.cluster(merge='sessions'))
	#~ print(a.cluster(merge='experiments'))
	#~ print(a.cluster(merge='names'))
	#~ a.scatter_parameters(merge='subjects')
	#~ a.scatter_parameters(merge='sessions')
	#~ plt.show(True)

if __name__=="__main__":
	test()
