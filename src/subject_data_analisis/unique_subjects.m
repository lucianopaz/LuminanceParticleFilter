function [subjects] = unique_subjects(data_dir)
% subjects = unique_subjects(data_dir)
% This function explores de data_dir supplied by the user and finds the
% unique subjects that participated in the experiment. The output is a
% structure array. Each array element contains a single subject's
% information:
%    name:       the subject's name
%    id:         a numeric id index
%    blocks:     an array with the experiment block index
%    data_files: the .mat files path with the subject's experimental data.
%                The path is not absolute, but relative to the supplied
%                data_dir.
%    nsessions:  The number of sessions done by the subject.

% Find the .mat files in data_dir. Expects a typical filename format
files = dir(fullfile(data_dir,'*.mat'));
file_names = {files.name};

% In this loop, the function extracts the subjects name and the
% experimental block from the filename.
sujs = cell(length(file_names),1);
blocks = zeros(length(file_names),1);
for i=1:length(file_names)
    name = file_names{i};
    slashes = strfind(name,'_');
    sujs{i} = lower(name(slashes(2)+2:slashes(3)-1));
    blocks(i) = str2double(name(slashes(3)+2:slashes(4)-1));
end

% List the unique subjects
[unique_suj,bla,c]=unique(sujs);

% Initiate the output structure
subjects = struct('name',unique_suj,'id',num2cell(1:length(unique_suj))',...
    'blocks',[],'data_files',[],'nsessions',NaN);
% The output structure data
for i = 1:length(subjects)
    inds = find(c==i);
    % The data_files will be sorted according to the experimental block
    % index.
    [subjects(i).blocks,ix] = sort(blocks(inds));
    subjects(i).data_files = cellfun(@(x) fullfile(data_dir,x),file_names(inds(ix)),...
                                     'UniformOutput',0);
    subjects(i).nsessions = length(inds);
end