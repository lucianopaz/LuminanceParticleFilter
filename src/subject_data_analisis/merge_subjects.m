function sm = merge_subjects(s)
% Merge subject data.
% Syntax:
% merge_subject = merge_subjects(subject_array)
% 
% Merges the supplied subject array into a single subject struct
    sm = s(1);
    sm.name = 'all';
    sm.id = 0;
    b = [];
    d = {};
    ns = [];
    for i = 1:length(s)
        b = [b;s(i).blocks];
        d = [d,s(i).data_files];
        ns = [ns;s(i).nsessions];
    end
    [sm.blocks,ix] = sort(b);
    sm.data_files = d(ix);
    sm.nsessions = max(ns);
end