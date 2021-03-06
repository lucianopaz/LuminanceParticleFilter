#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package for loading the behavioral dataset """

from __future__ import division
import numpy as np
from scipy import io as io
import os, itertools, sys, random

class subject:
	def __init__(self,name,id,blocks,data_files,nsessions):
		self.name = name
		self.id = id
		self.blocks = blocks
		self.data_files = data_files
		self.nsessions = nsessions
	def column_description(self):
		return column_description()
	def iter_data(self,max_block=np.Inf):
		for b,d in itertools.izip(self.blocks,self.data_files):
			if b>max_block:
				continue
			aux = io.loadmat(d);
			target = np.transpose(np.array([s[0] for s in aux['stim'].squeeze()]),(0,2,1))
			distractor = np.transpose(np.array([s[1] for s in aux['stim'].squeeze()]),(0,2,1))
			mean_target_lum = aux['trial'][:,1]
			rt = aux['trial'][:,5]-200 # Substract 200ms from recorded response time because fixation duration was not substracted during the experimental data acquisition
			performance = aux['trial'][:,7] # 1 for success, 0 for fail
			confidence = aux['trial'][:,8] # 2 for high confidence, 1 for low confidence
			if aux['trial'].shape[1]>9:
				selected_side = aux['trial'][:,9];
			else:
				selected_side = np.nan*np.ones_like(rt)
			data_matrix = np.array([mean_target_lum,rt,performance,confidence,selected_side,
				self.id*np.ones_like(rt),b*np.ones_like(rt)]).T
			yield data_matrix,target,distractor
	def load_data(self,max_block=np.Inf):
		first_element = True
		for data_matrix,target,distractor in self.iter_data(max_block=max_block):
			if first_element:
				all_data = data_matrix
				all_target = target
				all_distractor = distractor
			else:
				all_data = np.concatenate((all_data,data_matrix),axis=0)
				all_target = np.concatenate((all_target,target),axis=0)
				all_distractor = np.concatenate((all_distractor,distractor),axis=0)
			first_element = False
		return all_data,all_target,all_distractor
	def copy(self):
		return subject(self.name,self.id,self.blocks.copy(),self.data_files.copy(),self.nsessions)

def column_description():
	out = ['mean target lum','RT','performance','confidence','selected side','subject id','experiment block']
	print out
	return out

def unique_subjects(data_dir):
	""" subjects = unique_subjects(data_dir)
	 This function explores de data_dir supplied by the user and finds the
	 unique subjects that participated in the experiment. The output is a
	 list of subjects. """
	# Find the .mat files in data_dir. Expects a typical filename format
	files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
	
	# In this loop, the function extracts the subjects name and the
	# experimental block from the filename.
	file_suj_block = []
	blocks = []
	unique_sujs = set([])
	for f in files:
		temp = f.split("_")
		file_suj_block.append((f,temp[2][1:],int(temp[3][1:])))
		unique_sujs.add(temp[2][1:])
	
	# Initiate the output structure
	output = []
	id = 0
	for s in unique_sujs:
		id+=1
		suj_files = [(f,suj,block) for f,suj,block in file_suj_block if suj==s]
		data_files = [data_dir+"/"+f for f,suj,block in suj_files]
		blocks = sorted([block for f,suj,block in suj_files])
		nsessions = len(suj_files)
		output.append(subject(s,id,blocks,data_files,nsessions))
	return output

def merge_subjects(subject_list,name='all',subject_id=0):
	for i,s in enumerate(subject_list):
		if i==0:
			b = [bla for bla in s.blocks]
			d = [bla for bla in s.data_files]
			ns = s.nsessions
		else:
			b.extend(s.blocks);
			d.extend(s.data_files)
			ns+= s.nsessions
	ix = np.argsort(b)
	return subject(name,subject_id,[b[i] for i in ix],[d[i] for i in ix],ns)

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

def test(data_dir='/Users/luciano/Facultad/datos'):
	try:
		from matplotlib import pyplot as plt
		loaded_plot_libs = True
	except:
		loaded_plot_libs = False
	
	subjects = unique_subjects(data_dir)
	print str(len(subjects))+' subjects found'
	ms = merge_subjects(subjects)
	print 'Successfully merged all subjects with a total of '+str(ms.nsessions)+' sessions'
	dat,t,d = ms.load_data()
	print 'Loaded all data. Printing matrices shapes'
	print dat.shape, t.shape, d.shape
	print 'Data from '+str(dat.shape[0])+' trials loaded'
	column_content = column_description()
	print dat[0:10,:]
	
	print 'Testing increase_histogram'
	
	n = dat.shape[0]*2+10
	targetmean,indeces = increase_histogram_count(dat[:,0],n)
	ud, histogram = np.unique(dat[:,0], return_counts=True)
	inc_ud, inc_histogram = np.unique(targetmean, return_counts=True)
	print 'Increased histogram and indeces match?'
	print np.all(targetmean==dat[indeces,0])
	print 'Does the increase histogram have the desired number of elements?'
	print np.sum(inc_histogram)==n
	if loaded_plot_libs:
		# Plot histograms
		plt.figure(figsize=(13,10))
		# Decision kernel locked on stimulus onset
		plt.bar(ud-0.45,histogram/np.sum(histogram),width=0.45, bottom=0,color='b')
		plt.bar(inc_ud,inc_histogram/np.sum(inc_histogram),width=0.45, bottom=0,color='r')
		plt.xlabel('Unique luminance values [$cd/m^{2}$]')
		plt.ylabel('Fraction of ocurrences')
		plt.legend(['Raw data','Increased histogram'])
		plt.show()
	else:
		print 'Unique luminance raw data and increased histogram data'
		print ud
		print inc_ud
		print 'Number of ocurrences in raw data'
		print histogram, np.sum(histogram)
		print 'Number of ocurrences in increased histogram'
		print inc_histogram, np.sum(inc_histogram)

if __name__=="__main__":
	if len(sys.argv)>1:
		test(sys.argv[1])
	else:
		test()
