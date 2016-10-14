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
	def __init__(self,method = 'full_confidence', optimizer = 'cma', suffix = '', override=False,\
				n_clusters=2, affinity='euclidean', linkage='ward', pooling_func=np.nanmean, connectivity=None):
		self.method = method
		self.optimizer = optimizer
		self.suffix = suffix
		self.get_summary(override=override)
		self.init_clusterer(n_clusters=n_clusters, affinity=affinity,\
				linkage=linkage, pooling_func=pooling_func, connectivity=connectivity)
	
	def init_clusterer(self,n_clusters=2, affinity='euclidean', compute_full_tree=True,\
						linkage='ward', pooling_func=np.nanmean,connectivity=None):
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
		full_confidence_merit = fitter.forced_compute_full_confidence_merit(parameters)
		full_merit = fitter.forced_compute_full_merit(parameters)
		confidence_only_merit = fitter.forced_compute_confidence_only_merit(parameters)
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
		
		key = 'experiment_'+fitter.experiment+'_subject_'+fitter.subjectSession.get_name()+'_session_'+fitter.subjectSession.get_session()
		out = {key:{'experiment':fitter.experiment,'parameters':parameters,'full_merit':full_merit,\
					'full_confidence_merit':full_confidence_merit,'confidence_only_merit':confidence_only_merit,\
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
				if pn=='high_confidence_threshold':
					val = pd[pn] if pd[pn]<=2. else np.nan
				elif pn=='confidence_map_slope':
					val = pd[pn] if pd[pn]<=100. else np.nan
				else:
					val = pd[pn]
				vals.append(val)
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
	
	def controlled_scatter(self,scattered_parameters=['cost','internal_var','phase_out_prob'],axes=None,color_category='experiment',marker_category='session',merge=None,show=False):
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

def cluster_analysis(method='full_confidence', optimizer='cma', suffix='', override=False,\
				n_clusters=2, affinity='euclidean', linkage='ward', pooling_func=np.nanmean,\
				merge='names',filter_nans='post', tree_mode='r',show=False,extension='svg'):
	a = Analyzer(method, optimizer, suffix, override, n_clusters, affinity, linkage, pooling_func)
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
	
	if show:
		a.controlled_scatter(scattered_parameters=decision_parameters,merge=merge)
		plt.legend(loc='best', fancybox=True, framealpha=0.5)
		a.controlled_scatter(scattered_parameters=confidence_parameters,merge=merge)
		plt.legend(loc='best', fancybox=True, framealpha=0.5)
		plt.show(True)

def parameter_correlation(method='full_confidence', optimizer='cma', suffix='', override=False,\
				n_clusters=2, affinity='euclidean', linkage='ward', pooling_func=np.nanmean,\
				merge='names',filter_nans='post', tree_mode='r',show=False,extension='svg'):
	a = Analyzer(method, optimizer, suffix, override, n_clusters, affinity, linkage, pooling_func)
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
	
	dtype = [('parameters','O'),('name','i'),('session','i'),('experiment',experiments.dtype)]
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
			'all': copy.copy(parameter_names),\
			#~ 'all': ['cost','internal_var','phase_out_prob','high_confidence_threshold','confidence_map_slope'],\
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
		correct_rho_pval(pval1)
		c1[pval1>0.05] = np.nan
		c1s.append(c1)
		
		# Test pearson correlation treating the parameters as categories
		p2 = temp.reshape((-1,temp.shape[-1])).T
		c2,pval2 = utils.corrcoef(p2,method='pearson')
		#~ np.fill_diagonal(c2, np.nan)
		#~ np.fill_diagonal(pval2, np.nan)
		correct_rho_pval(pval2)
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
	ps = []
	for rowind,row in enumerate(pvals):
		for pval in pvals[rowind+1:]:
			ps.append(pval)
	ps = utils.holm_bonferroni(np.array(ps))
	counter = 0
	for j,pj in enumerate(pvals):
		for k,pk in enumerate(pvals[j+1:]):
			pvals[j,j+k+1] = ps[counter]
			pvals[j+k+1,j] = ps[counter]
			counter+=1

def binary_confidence_analysis(method='full_confidence', optimizer='cma', suffix='', override=False,\
				n_clusters=2, affinity='euclidean', linkage='ward', pooling_func=np.nanmean,\
				merge='names',filter_nans='post', tree_mode='r',show=False,extension='svg'):
	a = Analyzer(method, optimizer, suffix, override, n_clusters, affinity, linkage, pooling_func)
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

def test():
	parameter_correlation()
	return
	binary_confidence_analysis()
	
	a = Analyzer()
	a.get_parameter_array_from_summary(normalize={'internal_var':'experiment','dead_time':'name','dead_time_sigma':'session'})
	#~ tree = a.cluster(merge='names')
	#~ tree.copy().render('cluster_test.svg',tree_style=default_tree_style(mode='r'), layout=default_tree_layout)
	#~ tree.show(tree_style=default_tree_style(mode='r'))
	#~ tree = a.cluster(merge='sessions')
	#~ tree.copy().render('cluster_test.svg',tree_style=default_tree_style(mode='c'), layout=default_tree_layout)
	#~ tree.show(tree_style=default_tree_style(mode='c'))
	#~ tree = a.cluster()
	#~ tree.copy().render('cluster_test.svg',tree_style=default_tree_style(mode='c'), layout=default_tree_layout)
	#~ tree.show(tree_style=default_tree_style(mode='c'))
	#~ a.scatter_parameters()
	#~ a.scatter_parameters(merge='subjects')
	#~ a.scatter_parameters(merge='sessions')
	#~ a.set_pooling_func(np.median)
	#~ a.scatter_parameters(merge='subjects')
	#~ a.scatter_parameters(merge='sessions')
	unams,indnams = np.unique(a._names,return_inverse=True)
	#~ uexps,indexps = np.unique(a._experiments,return_inverse=True)
	usess,indsess = np.unique(a._sessions,return_inverse=True)
	
	for un in unams:
		inds = a._names==un
		pars = a._parameters[inds]
		cost = pars[:,a._parameter_names.index('cost')]
		internal_var = pars[:,a._parameter_names.index('internal_var')]
		phase_out_prob = pars[:,a._parameter_names.index('phase_out_prob')]
		high_confidence_threshold = pars[:,a._parameter_names.index('high_confidence_threshold')]
		confidence_map_slope = pars[:,a._parameter_names.index('confidence_map_slope')]
		plt.figure(figsize=(10,10))
		ax1 = plt.subplot(221)
		ax2 = plt.subplot(222)
		ax3 = plt.subplot(223)
		ax4 = plt.subplot(224)
		for us,marker in zip(usess,['o','s','D']):
			inds2 = a._sessions[inds]==us
			if any(inds2):
				colors = [{'2AFC':'r','Auditivo':'g','Luminancia':'b'}[str(x).strip('\x00')] for x in a._experiments[inds][inds2]]
				ax1.scatter(cost[inds2],internal_var[inds2],c=colors,s=20,label=us,marker=marker)
				ax2.scatter(cost[inds2],phase_out_prob[inds2],c=colors,s=20,label=us,marker=marker)
				ax3.scatter(internal_var[inds2],phase_out_prob[inds2],c=colors,s=20,label=us,marker=marker)
				ax4.scatter(confidence_map_slope[inds2],high_confidence_threshold[inds2],c=colors,s=20,label=us,marker=marker)
		ax1.set_xlabel('cost')
		ax1.set_ylabel('internal_var')
		ax2.set_xlabel('cost')
		ax2.set_ylabel('phase_out_prob')
		ax3.set_xlabel('internal_var')
		ax3.set_ylabel('phase_out_prob')
		ax3.legend()
		ax4.set_xlabel('confidence_map_slope')
		ax4.set_ylabel('high_confidence_threshold')
		plt.suptitle('Subject: '+str(un))
		plt.show(True)

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
