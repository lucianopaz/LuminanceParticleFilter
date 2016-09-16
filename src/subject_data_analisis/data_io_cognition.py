#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package for loading the behavioral dataset """

from __future__ import division
import numpy as np
from scipy import io as io
import os, itertools, sys, random, re, scipy.integrate

class SubjectSession:
	def __init__(self,name,session,experiment,data_dir):
		self.name = str(name)
		try:
			self.session = int(session)
			self._single_session = True
		except:
			self.session = [int(s) for s in session]
			self._single_session = False
		self.experiment = str(experiment)
		if isinstance(data_dir,list):
			self.data_dir = [str(d) for d in data_dir]
			self._single_data_dir = False
		else:
			self.data_dir = str(data_dir)
			self._single_data_dir = True
	
	def get_session(self):
		if self._single_session:
			session = str(self.session)
		else:
			session = '['+'-'.join([str(s) for s in self.session])+']'
		return session
	
	def get_key(self):
		return '{experiment}_name={name}_session={session}'.format(experiment=self.experiment,name=self.name,session=self.get_session())
	
	def list_data_files(self,override_raw_data_dir=None):
		if self._single_data_dir:
			if override_raw_data_dir:
				data_dir = self.data_dir.replace(override_raw_data_dir['original'],override_raw_data_dir['replacement'])
			else:
				data_dir = self.data_dir
			return [os.path.join(data_dir,f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir,f))]
		else:
			listdirs = []
			for dd in self.data_dir:
				if override_raw_data_dir:
					dd = dd.replace(override_raw_data_dir['original'],override_raw_data_dir['replacement'])
				listdirs.extend([os.path.join(dd,f) for f in os.listdir(dd) if os.path.isfile(os.path.join(dd,f))])
			return list(set(listdirs))
	
	def iter_data(self,override_raw_data_dir=None):
		if self.experiment=='Luminancia':
			if self._single_session:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if ((int(re.search('(?<=_B)[0-9]+(?=_)',f).group())-1)//4+1)==self.session and f.endswith('.mat')]
			else:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if ((int(re.search('(?<=_B)[0-9]+(?=_)',f).group())-1)//4+1) in self.session and f.endswith('.mat')]
			sessions = [((int(re.search('(?<=_B)[0-9]+(?=_)',f).group())-1)//4+1) for f in data_files]
			for f,session in zip(data_files,sessions):
				aux = io.loadmat(f)
				mean_target_lum = aux['trial'][:,1]
				rt = aux['trial'][:,5]*1e-3 # Convert to seconds
				performance = aux['trial'][:,7] # 1 for success, 0 for fail
				confidence = aux['trial'][:,8] # 2 for high confidence, 1 for low confidence
				if aux['trial'].shape[1]>9:
					selected_side = aux['trial'][:,9];
				else:
					selected_side = np.nan*np.ones_like(rt)
				if isinstance(self.name,int):
					data_matrix = np.array([mean_target_lum,rt,performance,confidence,selected_side,
										self.name*np.ones_like(rt),session*np.ones_like(rt)]).squeeze().T
				else:
					data_matrix = np.array([mean_target_lum,rt,performance,confidence,selected_side,
										session*np.ones_like(rt)]).squeeze().T
				yield data_matrix
		elif self.experiment=='2AFC':
			if self._single_session:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if int(re.search('(?<=sesion)[0-9]+',f).group())==self.session and f.endswith('.txt')]
			else:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if int(re.search('(?<=sesion)[0-9]+',f).group()) in self.session and f.endswith('.txt')]
			sessions = [int(re.search('(?<=sesion)[0-9]+',f).group()) for f in data_files]
			for f,session in zip(data_files,sessions):
				selected_side, performance, rt, contraste, confidence, phase, orientation = np.loadtxt(f, delimiter=' ', unpack=True)
				if isinstance(self.name,int):
					data_matrix = np.array([contraste,rt,performance,confidence,selected_side,
										orientation,phase,self.name*np.ones_like(rt),session*np.ones_like(rt)]).squeeze().T
				else:
					data_matrix = np.array([contraste,rt,performance,confidence,selected_side,
										orientation,phase,session*np.ones_like(rt)]).squeeze().T
				yield data_matrix
		elif self.experiment=='Auditivo':
			if self._single_session:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if int(re.search('(?<=sesion)[0-9]+',f).group())==self.session and not f.endswith('quest.mat') and f.endswith('.mat')]
			else:
				data_files = [f for f in self.list_data_files(override_raw_data_dir) if int(re.search('(?<=sesion)[0-9]+',f).group()) in self.session and not f.endswith('quest.mat') and f.endswith('.mat')]
			sessions = [int(re.search('(?<=sesion)[0-9]+',f).group()) for f in data_files]
			for f,session in zip(data_files,sessions):
				aux = io.loadmat(f)
				contraste = aux['QQ']
				rt = aux['RT']
				performance = aux['correct']
				confidence = aux['SEGU']+0.5
				confidence[confidence>1.] = 1.
				confidence[confidence<0.] = 0.
				selected_side = aux['RTA']
				confidence_rt = aux['SEGUTIME']
				target_location = aux['orden']
				if isinstance(self.name,int):
					data_matrix = np.array([contraste,rt,performance,confidence,selected_side,
										confidence_rt,target_location,self.name*np.ones_like(rt),session*np.ones_like(rt)]).squeeze().T
				else:
					data_matrix = np.array([contraste,rt,performance,confidence,selected_side,
										confidence_rt,target_location,session*np.ones_like(rt)]).squeeze().T
				yield data_matrix
	
	def load_data(self,override_raw_data_dir=None):
		first_element = True
		for data_matrix in self.iter_data(override_raw_data_dir=override_raw_data_dir):
			if first_element:
				all_data = data_matrix
				first_element = False
			else:
				all_data = np.concatenate((all_data,data_matrix),axis=0)
		return all_data
	
	def column_description(self):
		numeric_name = isinstance(self.name,int)
		if self.experiment=='Luminancia':
			if numeric_name:
				return ['mean target lum [cd/m^2]','RT [s]','performance','confidence','selected side','subject id','session']
			else:
				return ['mean target lum [cd/m^2]','RT [s]','performance','confidence','selected side','session']
		elif self.experiment=='2AFC':
			if numeric_name:
				return ['contraste','RT [s]','performance','confidence','selected side','orientation [º]','phase','subject id','session']
			else:
				return ['contraste','RT [s]','performance','confidence','selected side','orientation [º]','phase','session']
		elif self.experiment=='Auditivo':
			if numeric_name:
				return ['contraste','RT [s]','performance','confidence','selected side','confidence RT [s]','target location','subject id','session']
			else:
				return ['contraste','RT [s]','performance','confidence','selected side','confidence RT [s]','target location','session']
		else:
			raise ValueError('No column description available for the experiment: {0}'.format(self.experiment))
	
	def __getstate__(self):
		return {'name':self.name,'session':self.session,'_single_session':self._single_session,\
				'experiment':self.experiment,'_single_data_dir':self._single_data_dir,'data_dir':self.data_dir}
	
	def __setstate__(self,state):
		self.name = state['name']
		self.session = state['session']
		self._single_session = state['_single_session']
		self.experiment = state['experiment']
		self._single_data_dir = state['_single_data_dir']
		self.data_dir = state['data_dir']

def unique_subject_sessions(raw_data_dir,filter_by_experiment=None,filter_by_session=None):
	"""
	subjects = unique_subjects(raw_data_dir,filter_by_experiment=None,filter_by_session=None)
	 This function explores de data_dir supplied by the user and finds the
	 unique subjects that participated in the experiment. The output is a
	 list of subjects.
	 """
	output = []
	experiments = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir,d))]
	for experiment in experiments:
		# We want the data of all experiments except sperling's
		if experiment=='sperling':
			continue
		if filter_by_experiment:
			if experiment!=filter_by_experiment:
				continue
		
		experiment_data_dir = os.path.join(raw_data_dir,experiment,'COMPLETOS')
		subject_rel_dirs = sorted([d for d in os.listdir(experiment_data_dir) if os.path.isdir(os.path.join(experiment_data_dir,d))])
		for subject_rel_dir in subject_rel_dirs:
			name = subject_rel_dir.lower()
			subject_dir = os.path.join(experiment_data_dir,subject_rel_dir)
			files = os.listdir(subject_dir)
			if experiment!='Luminancia':
				sessions = sorted(list(set([int(re.search('(?<=sesion)[0-9]+',f).group()) for f in files])))
			else:
				blocks = sorted(list(set([int(re.search('(?<=_B)[0-9]+(?=_)',f).group()) for f in files])))
				sessions = sorted(list(set([(block-1)//4+1 for block in blocks])))
			for session in sessions:
				if filter_by_session:
					if session!=filter_by_session:
						continue
				output.append(SubjectSession(name,session,experiment,subject_dir))
	return anonimize_subjects(output)

def anonimize_subjects(subjectSessions):
	"""
	subjectSessions = anonimize_subjects(subjectSessions)
	 Takes a list of SubjectSession objects and converts their names into
	 a numerical id that overrides their original names.
	"""
	names = sorted(list(set([ss.name for ss in subjectSessions])))
	name_to_id = {}
	for subject_id,name in enumerate(names):
		name_to_id[name] = subject_id
	for ss in subjectSessions:
		ss.name = name_to_id[ss.name]
	return subjectSessions

def filter_subjects_list(subjectSessions,criteria='all_experiments',filter_details=None):
	output = []
	if criteria=='all_experiments':
		names = [s.name for s in subjectSessions]
		experiments = [s.experiment for s in subjectSessions]
		unique_names = sorted(list(set(names)))
		n_experiments = len(set(experiments))
		for name in unique_names:
			if len(set([e for n,e in zip(names,experiments) if n==name]))==n_experiments:
				output.extend([s for s in subjectSessions if s.name==name])
	elif criteria=='all_sessions_by_experiment':
		names = [s.name for s in subjectSessions]
		experiments = [s.experiment for s in subjectSessions]
		sessions = [s.session for s in subjectSessions]
		unique_names = sorted(list(set(names)))
		unique_experiments = sorted(list(set(experiments)))
		n_sessions = [len(set([s for s,e in zip(sessions,experiments) if e==ue])) for ue in unique_experiments]
		for name in unique_names:
			satifies_filter = True
			for i,experiment in enumerate(unique_experiments):
				if len(set([s for s,n,e in zip(sessions,names,experiments) if n==name and e==experiment]))!=n_sessions[i]:
					satifies_filter = False
					break
			if satifies_filter:
				output.extend([s for s in subjectSessions if s.name==name])
	elif criteria.startswith('experiment'):
		experiment = criteria.split('_')[1]
		output = [s for s in subjectSessions if s.experiment==experiment]
	else:
		raise ValueError('The specified criteria: "{0}" is not implemented'.format(criteria))
	return output

def merge_data_by_experiment(subjectSessions,filter_by_experiment=None,filter_by_session=None,return_column_headers=False):
	unique_experiments = sorted(list(set([s.experiment for s in subjectSessions])))
	output = {}
	if return_column_headers:
		output['headers'] = {}
	for experiment in unique_experiments:
		if filter_by_experiment:
			if experiment!=filter_by_experiment:
				continue
		output[experiment] = None
		if return_column_headers:
			output['headers'][experiment] = None
		merged_data = None
		for s in (s for s in subjectSessions if s.experiment==experiment):
			if filter_by_session:
				if s.session!=filter_by_session:
					continue
			data = s.load_data()
			if merged_data is None:
				merged_data = data
			else:
				merged_data = np.vstack((merged_data,data))
			if return_column_headers:
				if output['headers'][experiment] is None:
					output['headers'][experiment] = s.column_description()
		output[experiment] = merged_data
	return output

def merge_subjectSessions(subjectSessions,merge='all'):
	merge = merge.lower()
	if merge=='all':
		data_dirs = {}
		sessions = {}
		for ss in subjectSessions:
			exp = ss.experiment
			if exp not in data_dirs.keys():
				data_dirs[exp] = []
			if ss._single_data_dir:
				data_dirs[exp].append(ss.data_dir)
			else:
				data_dirs[exp].extend(ss.data_dir)
			if exp not in sessions.keys():
				sessions[exp] = []
			if ss._single_session:
				sessions[exp].append(ss.session)
			else:
				sessions[exp].extend(ss.session)
		output = [SubjectSession('all',sessions[exp],exp,data_dirs[exp]) for exp in data_dirs.keys()]
	elif merge=='sessions':
		sessions = {}
		for ss in subjectSessions:
			exp = str(ss.experiment)
			name = str(ss.name)
			data_dir = ss.data_dir
			key = exp+'_'+name
			if key not in sessions.keys():
				sessions[key] = {'data':[],'name':name,'experiment':exp,'data_dir':data_dir}
			if ss._single_session:
				sessions[key]['data'].append(ss.session)
			else:
				sessions[key]['data'].extend(ss.session)
		output = [SubjectSession(sessions[key]['name'],sessions[key]['data'],sessions[key]['experiment'],sessions[key]['data_dir']) for key in sessions.keys()]
	elif merge=='subjects':
		data_dirs = {}
		for ss in subjectSessions:
			exp = str(ss.experiment)
			session = ss.session
			data_dir = ss.data_dir
			key = exp+'_'+str(session)
			if key not in data_dirs.keys():
				data_dirs[key] = {'data':[],'session':session,'experiment':exp}
			if ss._single_data_dir:
				data_dirs[key]['data'].append(ss.data_dir)
			else:
				data_dirs[key]['data'].extend(ss.data_dir)
		output = [SubjectSession('all',data_dirs[key]['session'],data_dirs[key]['experiment'],data_dirs[key]['data']) for key in data_dirs.keys()]
	else:
		ValueError('Unknown merge criteria "{0}"'.format(merge))
	return output

def increase_histogram_count(d,n):
	"""
	(out,indexes)=increase_histogram_count(d,n)
	
	Take an numpy.array of data d of shape (m,) and return an array out
	of shape (n,) that copies the elements of d keeping d's histogram
	approximately invariant. Second output "indexes" is an array so
	that out = d(indexes)
	"""
	d = d.squeeze()
	if len(d.shape)>1:
		raise(ValueError('Input data must be an array with only one dimension'))
	if n<len(d):
		raise(ValueError('n must be larger than the length of the data'))
	ud, ui, histogram = np.unique(d, return_inverse=True, return_counts=True)
	increased_histogram = np.floor(histogram*n/len(d))
	
	if np.sum(increased_histogram)<n:
		temp = np.zeros_like(histogram)
		cumprob = np.cumsum(histogram)/sum(histogram)
		for i in range(int(n-sum(increased_histogram))):
			ind = np.searchsorted(cumprob,random.random(),'left')
			temp[ind]+=1
		increased_histogram+=temp
	
	unique_indexes = []
	for i in range(len(ud)):
		unique_indexes.append(np.random.permutation([j for j,uii in enumerate(ui) if uii==i]))
	
	out = np.zeros(n)
	indexes = np.zeros_like(out,dtype=np.int)
	count_per_value = np.zeros_like(increased_histogram)
	for c in range(n):
		cumprob = np.cumsum(increased_histogram-count_per_value)/sum(increased_histogram-count_per_value)
		ind = np.searchsorted(cumprob,random.random(),'left')
		out[c] = ud[ind]
		indexes[c] = np.int(unique_indexes[ind][count_per_value[ind]%histogram[ind]])
		count_per_value[ind]+=1
	randperm_indexes = np.random.permutation(n)
	return out[randperm_indexes], indexes[randperm_indexes]

def compute_roc(performance,confidence,partition=101):
	edges = np.linspace(0,1,int(partition))
	roc = np.zeros((int(partition),2),dtype=np.float)
	hit = performance==1
	miss = np.logical_not(hit)
	nhits = np.sum(hit.astype(np.float))
	nmisses = np.sum(miss.astype(np.float))
	for i,ue in enumerate(edges):
		if i<int(partition)-1:
			p1 = np.sum(np.logical_and(confidence<ue,hit).astype(np.float))/nhits
			p2 = np.sum(np.logical_and(confidence<ue,miss).astype(np.float))/nmisses
		else:
			p1 = np.sum(np.logical_and(confidence<=ue,hit).astype(np.float))/nhits
			p2 = np.sum(np.logical_and(confidence<=ue,miss).astype(np.float))/nmisses
		roc[i,:] = np.array([p1,p2])
	return roc

def compute_auc(roc):
	return scipy.integrate.simps(roc[:,1],roc[:,0])

def test(raw_data_dir='/home/luciano/Dropbox/Luciano/datos joaquin/para_luciano/raw_data'):
	try:
		from matplotlib import pyplot as plt
		loaded_plot_libs = True
	except:
		loaded_plot_libs = False
	
	subjects = unique_subject_sessions(raw_data_dir)
	
	bla = {'2AFC':0.,'Auditivo':0.,'Luminancia':0.}
	for s in subjects:
		rt = np.sort(s.load_data()[:,1])
		key = s.experiment
		bla[key] = np.max([bla[key],np.std(rt)])
	print bla
	
	print str(len(subjects))+' subjectSessions can be constructed found'
	filtered_subjects = filter_subjects_list(subjects)
	print str(len(filtered_subjects))+' filtered subjectSessions with all_experiments criteria'
	subjects = filter_subjects_list(subjects,'all_sessions_by_experiment')
	print str(len(subjects))+' filtered subjectSessions with sessions_by_experiment criteria'
	
	merged_all = merge_subjectSessions(subjects,merge='all')
	print str(len(merged_all))+' merged subjectSessions with merge all'
	merged_sessions = merge_subjectSessions(subjects,merge='sessions')
	print str(len(merged_sessions))+' merged subjectSessions with merge sessions'
	merged_subjects = merge_subjectSessions(subjects,merge='subjects')
	print str(len(merged_subjects))+' merged subjectSessions with merge subjects'
	
	experiments_data = merge_data_by_experiment(subjects,return_column_headers=True)
	
	print 'Successfully merged all subjects data in '+str(len([k for k in experiments_data.keys() if k!='headers']))+' experiments'
	headers = experiments_data['headers']
	for key in experiments_data.keys():
		if key=='headers':
			continue
		data = experiments_data[key]
		matches = True
		for test_subj in [t for t in merged_all if t.experiment==key]:
			testdata = test_subj.load_data()
			test = testdata.shape[0]==data.shape[0]
			if not test:
				print testdata.shape,data.shape
			matches = matches and test
		print 'Merged all matches shape? {0}'.format('Yes' if matches else 'No')
		matches = True
		for test_subj in [t for t in merged_sessions if t.experiment==key]:
			testdata = test_subj.load_data()
			test = testdata.shape[0]==data[data[:,-2]==int(test_subj.name)].shape[0]
			if not test:
				print test_subj.name,testdata.shape,data[data[:,-2]==test_subj.name].shape
			matches = matches and test
		print 'Merged all sessions matches shape? {0}'.format('Yes' if matches else 'No')
		matches = True
		for test_subj in [t for t in merged_subjects if t.experiment==key]:
			testdata = test_subj.load_data()
			test = testdata.shape[0]==data[data[:,-1]==test_subj.session].shape[0]
			if not test:
				print testdata.shape,data[data[:,-1]==test_subj.session].shape
			matches = matches and test
		print 'Merged all subjects matches shape? {0}'.format('Yes' if matches else 'No')
		print '{0}: {1} trials, {2} sessions, {3} subjects'.format(key,data.shape[0],len(np.unique(data[:,-1])),len(np.unique(data[:,-2])))
		if loaded_plot_libs:
			inds = data[:,1]<14.
			plt.figure()
			plt.subplot(141)
			plt.hist(data[inds,0],100,normed=True)
			plt.xlabel(headers[key][0])
			plt.subplot(142)
			plt.hist(data[inds,1],100,normed=True)
			plt.xlabel(headers[key][1])
			plt.subplot(143)
			plt.hist(data[inds,2],2,normed=True)
			plt.xlabel(headers[key][2])
			plt.subplot(144)
			plt.hist(data[inds,3],100,normed=True)
			plt.xlabel(headers[key][3])
			plt.suptitle(key)
			
			roc = compute_roc(data[inds,2],data[inds,3])
			auc = compute_auc(roc)
			plt.figure()
			plt.plot(roc[:,0],roc[:,1])
			plt.xlabel(r'$P(conf<x|hit)$')
			plt.ylabel(r'$P(conf<x|miss)$')
			plt.title(key+' (AUC = {0})'.format(auc))
	plt.show(True)

if __name__=="__main__":
	if len(sys.argv)>1:
		test(sys.argv[1])
	else:
		test()
