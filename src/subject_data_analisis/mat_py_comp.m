datadir = fullfile('~','Dropbox','DecisionConfidenceKernels','data');
ISI = 40;

subjects = unique_subjects(datadir);
[data,target,distractor] = load_stim_and_trial(subjects,6);
tfluct = nan(size(target,1),125); tfluct(:,1:size(target,2)) = squeeze(mean(target,3))-repmat(data(:,1),1,size(target,2));
dfluct = nan(size(target,1),125); dfluct(:,1:size(target,2)) = squeeze(mean(distractor,3))-50;

target = repmat(squeeze(mean(target,3)),1,5);
tfluct(:,26:end) = randn(size(target,1),size(target,2)-25)*5;
target(:,26:end) = tfluct(:,26:end) + repmat(data(:,1),1,size(target,2)-25);
distractor = repmat(squeeze(mean(distractor,3)),1,5);
dfluct(:,26:end) = randn(size(distractor,1),size(distractor,2)-25)*5;
distractor(:,26:end) = dfluct(:,26:end) + 50;

pr_mu_t = 50;
pr_mu_d = 50;
pr_va_t = 15^2;
pr_va_d = 15^2;

prior_mu_t = pr_mu_t*ones(size(target));
prior_mu_d = pr_mu_d*ones(size(distractor));
prior_sigma_t = sqrt(pr_va_t)*ones(size(target));
prior_sigma_d = sqrt(pr_va_d)*ones(size(distractor));
sigma = 5;
n = repmat(1:size(target,2),size(target,1),1);

post_va_t = 1./(1./prior_sigma_t.^2+n./sigma.^2);
post_va_d = 1./(1./prior_sigma_d.^2+n./sigma.^2);
post_mu_t = (prior_mu_t./prior_sigma_t.^2+cumsum(target,2)/sigma.^2).*post_va_t;
post_mu_d = (prior_mu_d./prior_sigma_d.^2+cumsum(distractor,2)/sigma.^2).*post_va_d;

dprime = (post_mu_t-post_mu_d)./sqrt(post_va_t+post_va_d);
% dprime = post_mu_t./sqrt(post_va_t)-post_mu_d./sqrt(post_va_d);
% dprime = post_mu_t./post_va_t-post_mu_d./post_va_d;

threshold = 1.6;

threshold_passed = abs(dprime)>=threshold;
rt = zeros(size(dprime,1),1);
decided = false(size(rt));
performance = zeros(size(rt));
criterium = zeros(size(rt));
for i = 1:size(dprime,1)
    ind = find(threshold_passed(i,:),1);
    if ~isempty(ind)
        decided(i) = true;
        rt(i) = ind;
        if dprime(i,ind)>0
            performance(i) = 1;
        else
            performance(i) = 0;
        end
        criterium(i) = dprime(i,ind);
    else
        performance(i) = nan;
        criterium(i) = nan;
        rt(i) = size(dprime,2);
    end
end
rt = rt*ISI;

ret = [decided,performance,criterium,rt];
selection = performance;
selection(performance==0) = 2;
[dk,ck,dks,cks] = kernels(tfluct,dfluct,selection,ones(size(selection)));
% save('mat_py_comp.mat','ISI','sigma','target','distractor','threshold','pr_mu_t','pr_mu_d','pr_va_t','pr_va_d','ret','dprime','tfluct','dfluct','dk','dks');
%%
errorzone(dk(:,1:25)',dks(:,1:25)','edgealpha',0,'facealpha',0.3)