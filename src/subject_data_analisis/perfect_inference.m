function perfect_inference()

datadir = fullfile('~','Dropbox','DecisionConfidenceKernels','data');

subjects = unique_subjects(datadir);
[data,target,distractor] = load_stim_and_trial(subjects,6);
tfluct = squeeze(mean(target,3))-repmat(data(:,1),1,size(target,2));
dfluct = squeeze(mean(distractor,3))-50;
tfluct_ext = nan(size(tfluct,1),125); tfluct_ext(:,1:size(tfluct,2)) = tfluct;
dfluct_ext = nan(size(dfluct,1),125); dfluct_ext(:,1:size(dfluct,2)) = dfluct;

target = repmat(squeeze(mean(target,3)),1,5);
target(:,26:end) = randn(size(target,1),size(target,2)-25)*5 + repmat(data(:,1),1,size(target,2)-25);
distractor = repmat(squeeze(mean(distractor,3)),1,5);
distractor(:,26:end) = randn(size(distractor,1),size(distractor,2)-25)*5 + 50;

selection = data(:,3); selection(data(:,3)~=1) = 2;
confidence = data(:,4);

T_dec = data(:,2);
[bla,T_dec_ind] = histc(T_dec,0:40:5000);
T_dec_ind(T_dec_ind==126) = 125;

%%

prior_mu_t = 50*ones(size(target));
prior_mu_d = 50*ones(size(distractor));
prior_sigma_t = 15*ones(size(target));
prior_sigma_d = 15*ones(size(distractor));
sigma = 5;
n = repmat(1:size(target,2),size(target,1),1);

post_va_t = 1./(1./prior_sigma_t.^2+n./sigma.^2);
post_va_d = 1./(1./prior_sigma_d.^2+n./sigma.^2);
post_mu_t = (prior_mu_t./prior_sigma_t.^2+cumsum(target,2)/sigma.^2).*post_va_t;
post_mu_d = (prior_mu_d./prior_sigma_d.^2+cumsum(distractor,2)/sigma.^2).*post_va_d;

dprime = post_mu_t./post_va_t-post_mu_d./post_va_d;

T = (0:size(target,2))*40;
RT = data(:,2);

[fitted_vars,fval,exitflag,output,lambda,grad,hessian] = fmincon(@merit,[1.6,200],[],[],[],[],[0,0],[],[],optimset('tolfun',1e-10,'tolx',1e-10,'tolcon',1e-12));
covariance = inv(hessian);
disp(['Fitted threshold = ',num2str(fitted_vars(1)),'+-',num2str(sqrt(covariance(1,1)))])
disp(['Fitted fixed delay = ',num2str(fitted_vars(2)),'+-',num2str(sqrt(covariance(2,2)))])
disp(['Best fit objective function value = ',num2str(fval)])

[sdec,sRT,s_tdec_ind] = simulate_decision(fitted_vars(1),fitted_vars(2));
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

prob_acierto = zeros(size(RT));
for j = 1:length(RT)
    prob_acierto(j) = 1-normcdf(0,post_mu_t(s_tdec_ind(j))-post_mu_d(s_tdec_ind(j)),sqrt(post_va_t(s_tdec_ind(j))+post_va_d(s_tdec_ind(j))));
end
% sim_confidence = prob_acierto;
% sim_confidence = 1./(1+exp(-5*(prob_acierto-median(prob_acierto))));
sim_confidence = 1./(1+exp(0.004*(T(s_tdec_ind)-median(T(s_tdec_ind)))))';


[decision_kernel,confidence_kernel,decision_kernel_std,confidence_kernel_std] = ...
    kernels(tfluct,dfluct,selection,confidence);

[sim_decision_kernel,sim_confidence_kernel,sim_decision_kernel_std,sim_confidence_kernel_std] = ...
    kernels(tfluct,dfluct,sdec,sim_confidence,false);

figure
T_kern = 0:40:960;
subplot(1,2,1)
errorzone(T_kern,decision_kernel(1,:),decision_kernel_std(1,:),'--b','edgealpha',0,'facealpha',0.3);
hold on
errorzone(T_kern,decision_kernel(2,:),decision_kernel_std(2,:),'--r','edgealpha',0,'facealpha',0.3);
errorzone(T_kern,sim_decision_kernel(1,:),sim_decision_kernel_std(1,:),'b','edgealpha',0,'facealpha',0.3);
errorzone(T_kern,sim_decision_kernel(2,:),sim_decision_kernel_std(2,:),'r','edgealpha',0,'facealpha',0.3);
hold off
title('Decision kernel')
xlabel('T [ms]')
legend({'Subject D_{S}','Subject D_{N}','Simulation D_{S}','Simulation D_{N}'})

subplot(1,2,2)
errorzone(T_kern,confidence_kernel(1,:),confidence_kernel_std(1,:),'--b','edgealpha',0,'facealpha',0.3);
hold on
errorzone(T_kern,confidence_kernel(2,:),confidence_kernel_std(2,:),'--r','edgealpha',0,'facealpha',0.3);
errorzone(T_kern,sim_confidence_kernel(1,:),sim_confidence_kernel_std(1,:),'b','edgealpha',0,'facealpha',0.3);
errorzone(T_kern,sim_confidence_kernel(2,:),sim_confidence_kernel_std(2,:),'r','edgealpha',0,'facealpha',0.3);
hold off
title('Confidence kernel')
xlabel('T [ms]')
legend({'Subject C_{S}','Subject C_{N}','Simulation C_{S}','Simulation C_{N}'})


sim_T_dec = sRT;
[bla,sim_T_dec_ind] = histc(sim_T_dec,0:40:5000);
sim_T_dec_ind(sim_T_dec_ind==126) = 125;
target(sim_T_dec_ind==0,:) = nan;
distractor(sim_T_dec_ind==0,:) = nan;
sim_T_dec_ind(sim_T_dec_ind==0) = 125;

[decision_kernel,confidence_kernel,decision_kernel_std,confidence_kernel_std] = ...
    kernels(tfluct_ext,dfluct_ext,selection,confidence,true,false,T_dec_ind);

[sim_decision_kernel,sim_confidence_kernel,sim_decision_kernel_std,sim_confidence_kernel_std] = ...
    kernels(target-repmat(data(:,1),1,size(target,2)),distractor-50,sdec,sim_confidence,false,false,sim_T_dec_ind);

figure
T_kern = -4960:40:4960;
subplot(1,2,1)
errorzone(T_kern,decision_kernel(1,:),decision_kernel_std(1,:),'--b','edgealpha',0,'facealpha',0.3);
hold on
errorzone(T_kern,decision_kernel(2,:),decision_kernel_std(2,:),'--r','edgealpha',0,'facealpha',0.3);
errorzone(T_kern,sim_decision_kernel(1,:),sim_decision_kernel_std(1,:),'b','edgealpha',0,'facealpha',0.3);
errorzone(T_kern,sim_decision_kernel(2,:),sim_decision_kernel_std(2,:),'r','edgealpha',0,'facealpha',0.3);
hold off
title('Decision kernel')
xlabel('T - RT [ms]')
legend({'Subject D_{S}','Subject D_{N}','Simulation D_{S}','Simulation D_{N}'})

subplot(1,2,2)
errorzone(T_kern,confidence_kernel(1,:),confidence_kernel_std(1,:),'--b','edgealpha',0,'facealpha',0.3);
hold on
errorzone(T_kern,confidence_kernel(2,:),confidence_kernel_std(2,:),'--r','edgealpha',0,'facealpha',0.3);
errorzone(T_kern,sim_confidence_kernel(1,:),sim_confidence_kernel_std(1,:),'b','edgealpha',0,'facealpha',0.3);
errorzone(T_kern,sim_confidence_kernel(2,:),sim_confidence_kernel_std(2,:),'r','edgealpha',0,'facealpha',0.3);
hold off
title('Confidence kernel')
xlabel('T - RT [ms]')
legend({'Subject C_{S}','Subject C_{N}','Simulation C_{S}','Simulation C_{N}'})

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
function [sim_dec,sim_RT,tdec_ind] = simulate_decision(t,b)
    sim_dec = zeros(size(RT));
    sim_RT = zeros(size(RT));
    threshold_passed = abs(dprime)>=t;
    tdec_ind = zeros(size(RT));
    for i = 1:size(dprime,1)
        ind = find(threshold_passed(i,:),1);
        if ~isempty(ind)
            tdec_ind(i) = ind;
            sim_RT(i) = T(ind)+b;
            if dprime(i,ind)>0
                sim_dec(i) = 1;
            else
                sim_dec(i) = 2;
            end
        else
            sim_RT(i) = T(end)+b;
            tdec_ind(i) = length(T);
        end
    end
end
end