#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package for loading the behavioral dataset """

import numpy as np
import scipy, os, itertools

class subject:
	def __init__(self,name,id,blocks,data_files,nsessions):
		self.name = name
		self.id = id
		self.blocks = blocks
		self.data_files = data_files
		self.nsessions = nsessions
	def load_data(self,max_block=np.Inf,max_session=np.Inf):
		for b,d,s in itertools.izip(self.blocks,self.data_files,self.nsessions) if b<=max_block and s<=max_session:
			aux = scipy.io.loadmat(d);
			target = np.transpose(
        target = permute(cat(3,aux.stim.T),[3,2,1]);
        distractor = permute(cat(3,aux.stim.D),[3,2,1]);
        % Parse the trial data
        mean_target_lum = aux.trial(:,2);
        rt = aux.trial(:,6);
        performance = aux.trial(:,8);
        confidence = aux.trial(:,9);
        if size(aux.trial,2)>9 % If the selected side was recorded
            selected_side = aux.trial(:,10);
        else
            selected_side = nan(size(performance));
        end

def unique_subjects(data_dir):
""" subjects = unique_subjects(data_dir)
 This function explores de data_dir supplied by the user and finds the
 unique subjects that participated in the experiment. The output is a
 list of subjects. """
	# Find the .mat files in data_dir. Expects a typical filename format
	files = [file for file in os.listdir(data_dir) if file.endswith(".mat")]
	
	# In this loop, the function extracts the subjects name and the
	# experimental block from the filename.
	file_suj_block = []
	blocks = []
	unique_sujs = set([])
	for file in files:
		temp = file.split("_")
		suj_block.append((file,temp[2][1:],int(temp[3][1:])))
		unique_sujs.add(temp[2][1:])
	
	# Initiate the output structure
	output = []
	id = 0
	for s in unique_sujs:
		id+=1
		suj_files = [(file,suj,block) for file,suj,block in suj_block if suj==s]
		data_files = [data_dir+"/"+file for file,suj,block in suj_file]
		blocks = [file for file,suj,block in suj_file].sort()
		nsessions = len(suj_files)
		output.append(subject(s,id,blocks,data_files,nsessions))
	return output

def merge_subjects(subject_list,name='all',subject_id=0):
	b = [];
	d = [];
	ns = [];
	for s in subject_list:
		b.extend(s.blocks);
		d.extend(s.data_files)
		ns.extend(s.nsessions)
	ix = np.argsort(b)
	return subject(name,subject_id,b[ix],d[ix],ns[ix])


function [all_trial_data,all_targets,all_distractors] = load_stim_and_trial(subjects,maxblock)
% [all_trial_data,all_targets,all_distractors] = 
%     load_stim_and_trial(subjects)
%     load_stim_and_trial(subjects,maxblock)
% 
% This function accepts as input a subjects structure as the one returned
% by unique_subjects and an optional input, maxblock.
% It loads the .mat files contained in the subjects.data_files cell array
% and parses the data into 3 matrices. If maxblock is supplied, it loads
% the .mat files corresponding to a block less or equal to maxblock.
% 
% The function outputs:
%    all_trial_data:  A matrix with Nx7 elements. Each row corresponds to a
%                     single experimental trial. The first column is the
%                     target patch mean luminance (the distractor is always
%                     50cd/m^2). The second column is the subject's
%                     response time (ms). The third column is the trial
%                     performance (1==hit, 0==miss). The fourth column is
%                     the trial confidence (1==low, 2==high). The fifth
%                     column is the subjects selected side (1==left,
%                     2==right, NaN== Side not recorded). The sixth column
%                     is the subject numeric id. The seventh column is the
%                     experimental block.
%    
%    all_targets:     A matrix with NxMx4 elements. Contains the target
%                     patches luminances in (cd/m^2). M is the number of
%                     discrete luminance changes. The Inter Stimulus
%                     Interval (ISI) usually is equal to 40ms. The third
%                     dimension 4 elements are the luminances of each of
%                     the 4 luminous target patches.
%    
%    all_distractors: The same as all_targets but contains the distractor
%                     patches luminances.

% Default initialization for maxblock
if nargin<2
    maxblock = inf;
end

% Initialize output
all_targets = [];
all_distractors = [];
all_trial_data = [];

% Loop over subjects
for i = 1:length(subjects)
    subject = subjects(i);
    if ~isValid(subject)
        warning('load_sat:invalidSubject','Encountered invalid subject. Will ignore it.');
        continue
    end
    for j = 1:length(subject.data_files)
        if subject.blocks(j)>maxblock
            % If the block is greater than maxblock, break loop and proceed
            % to next subject.
            break
        end
        % Load the .mat file
        aux = load(subject.data_files{j});
        % Rearrange the data in aux.stim in order to return the desired
        % matrix
        target = permute(cat(3,aux.stim.T),[3,2,1]);
        distractor = permute(cat(3,aux.stim.D),[3,2,1]);
        % Parse the trial data
        mean_target_lum = aux.trial(:,2);
        rt = aux.trial(:,6);
        performance = aux.trial(:,8);
        confidence = aux.trial(:,9);
        if size(aux.trial,2)>9 % If the selected side was recorded
            selected_side = aux.trial(:,10);
        else
            selected_side = nan(size(performance));
        end
        % Concatenate the loaded data to the outputs
        all_targets = cat(1,all_targets,target);
        all_distractors = cat(1,all_distractors,distractor);
        all_trial_data = cat(1,all_trial_data,...
            [mean_target_lum,rt,performance,confidence,selected_side,...
            subject.id*ones(size(rt)),subject.blocks(j)*ones(size(rt))]);
    end
end


    function result = isValid(s)
        result = false;
        if isstruct(s);
            if isfield(s,'blocks') && isfield(s,'data_files') && isfield(s,'id')
                result = true;
            end
        end
    end
end
