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

T = (0:size(target,2))*40;
RT = data(:,2);

[fitted_vars,fval,exitflag,output,lambda,grad,hessian] = fmincon(@merit,[1.18,200],[],[],[],[],[0,0],[],[],optimset('tolfun',1e-10,'tolx',1e-10,'tolcon',1e-12));
covariance = inv(hessian);
disp(['Fitted threshold = ',num2str(fitted_vars(1)),'+-',num2str(sqrt(covariance(1,1)))])
disp(['Fitted fixed delay = ',num2str(fitted_vars(2)),'+-',num2str(sqrt(covariance(2,2)))])
disp(['Best fit objective function value = ',num2str(fval)])

[sdec,sRT] = simulate_decision(fitted_vars(1),fitted_vars(2));
hist_RT = histc(RT,T);
hist_sRT = histc(sRT,T);
cociente = size(target,1)*(T(2)-T(1));
logn_params = lognfit(RT);
logn_estim = lognpdf(T,logn_params(1),logn_params(2));
figure
% subplot(1,2,1)
plot(T,hist_RT/cociente,'b')
hold on
plot(T,hist_sRT/cociente,'r')
plot(T,logn_estim,'k')
hold off
xlabel('RT [ms]')
ylabel('Prob density [1/ms]')
legend({'Subject','Fitted simulation','Log normal'})
set(findall(gcf,'type','line'),'linewidth',2)
% subplot(1,2,2)
% for j = 0:10
%     theo_RT = theoretical_rt_distribution(5,200,50,50,15,15,5,50+j,50);
%     plot(T,theo_RT)
%     hold all
% end
% hold off

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
%     function prob = theoretical_rt_distribution(threshold,shift,pmu_t,pmu_d,ps_t,ps_d,sigma,mu_t,mu_d,T_max)
%         % Esta mal porque hay que encarar esto resolviendo una ecuaci?n
%         % maestra
%         if nargin<10
%             T_max = T(end);
%         end
%         T_temp = (0:40:T_max)';
%         prob = zeros(length(T_temp),1);
%         n = (1:size(prob,1))';
%         
%         dprime_mean = (pmu_t./ps_t.^2-pmu_d./ps_d.^2+n.*(mu_t-mu_d)/sigma.^2);
%         dprime_std = sqrt(2).*n;
%         cprob = 0;
%         for ii=1:length(prob)
%             pdown = normcdf(-threshold,dprime_mean(ii),dprime_std(ii));
%             pup = 1-normcdf(threshold,dprime_mean(ii),dprime_std(ii));
%             prob(ii) = (pdown+pup)*(1-cprob);
%             cprob = cprob + prob(ii);
%         end
%         shift_ind = floor(shift/40);
%         high_prop = mod(shift,40);
%         temp_mat = diag((1-high_prop)*ones(length(prob)-shift_ind,1),-shift_ind)+diag(high_prop*ones(length(prob)-shift_ind-1,1),-shift_ind-1);
%         prob = temp_mat*prob;
%     end
end