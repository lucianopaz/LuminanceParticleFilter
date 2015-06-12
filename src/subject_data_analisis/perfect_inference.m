function perfect_inference()

datadir = fullfile('~','Dropbox','DecisionConfidenceKernels','data');

subjects = unique_subjects(datadir);
[data,target,distractor] = load_stim_and_trial(subjects,6);

% Compute subject decision and confidence kernels
% Reduce the 4 luminous patches to a single patch.
target = repmat(squeeze(mean(target,3)),1,5);
distractor = repmat(squeeze(mean(distractor,3)),1,5);

%%

prior_mu_t = 50*ones(size(target));
prior_mu_d = 50*ones(size(distractor));
prior_sigma_t = 15*ones(size(target));
prior_sigma_d = 15*ones(size(distractor));
sigma = 5;
n = repmat(1:size(target,2),size(target,1),1);

post_sigma_t = 1./(1./prior_sigma_t.^2+n./sigma.^2);
post_sigma_d = 1./(1./prior_sigma_d.^2+n./sigma.^2);
post_mu_t = (prior_mu_t./prior_sigma_t.^2+cumsum(target,2)/sigma.^2).*post_sigma_t;
post_mu_d = (prior_mu_d./prior_sigma_d.^2+cumsum(distractor,2)/sigma.^2).*post_sigma_d;

dprime = post_mu_t./post_sigma_t-post_mu_d./post_sigma_d;

T = (0:size(target,2)-1)*40;
RT = data(:,2);

[fitted_vars,fval,exitflag,output,lambda,grad,hessian] = fmincon(@merit,[1.2,200],[],[],[],[],[0,0],[],[],optimset('tolfun',1e-10,'tolx',1e-10,'tolcon',1e-12));
covariance = inv(hessian);
disp(['Fitted threshold = ',num2str(fitted_vars(1)),'+-',num2str(sqrt(covariance(1,1)))])
disp(['Fitted fixed delay = ',num2str(fitted_vars(2)),'+-',num2str(sqrt(covariance(2,2)))])
disp(['Best fit objective function value = ',num2str(fval)])

[sdec,sRT] = simulate_decision(fitted_vars(1),fitted_vars(2));
[hist_RT,bins] = hist(RT,100);
hist_sRT = hist(sRT,bins);
figure
plot(bins,hist_RT,'b')
hold on
plot(bins,hist_sRT,'r')
hold off

function out = merit(x)
    sim_RT = zeros(size(RT));
    threshold_passed = abs(dprime)>=x(1);
    for i = 1:size(dprime,1)
        ind = find(threshold_passed(i,:),1);
        if ~isempty(ind)
            sim_RT(i) = T(ind);
        else
            sim_RT(i) = T(end);
        end
    end
    out = sum((RT-sim_RT-x(2)).^2);
end
function [sim_dec,sim_RT] = simulate_decision(t,b)
    sim_dec = zeros(size(RT));
    sim_RT = zeros(size(RT));
    threshold_passed = abs(dprime)>=t;
    for i = 1:size(dprime,1)
        ind = find(threshold_passed(i,:),1);
        if ~isempty(ind)
            sim_RT(i) = T(ind)+b;
            if dprime(ind)>0
                sim_dec(i) = 1;
            else
                sim_dec(i) = 2;
            end
        else
            sim_RT(i) = T(end)+b;
        end
    end
end
    function theoretical_rt_distribution(pmu_t,pmu_d,ps_t,ps_d,sigma,mu_t,mu_d)
        
    end
end