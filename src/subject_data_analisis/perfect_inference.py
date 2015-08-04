#!/usr/bin/python
#-*- coding: UTF-8 -*-
""" Package for fitting perfect inference behavior """

from __future__ import division
import numpy as np
import data_io as io
import kernels as ke
import os, itertools, sys, random

class PerfectInference():
	"""
	Class that implements perfect bayes inference on the mean luminance
	classification task with known model variance
	"""
	def __init__(self,model_sigma_t=1.,model_sigma_d=1.,prior_mu_t=0.,prior_mu_d=0.,prior_si_t=1.,prior_si_d=1.,\
				threshold=5.):
		self.sigma_t = model_sigma_t
		self.sigma_d = model_sigma_d
		self.prior_mu_t = prior_mu_t
		self.prior_mu_d = prior_mu_d
		self.prior_si_t = prior_si_t
		self.prior_si_d = prior_si_d
		
		self.threshold = threshold
		self.signals = []
	
	def onlineInference(self,targetSignal,distractorSignal,storeSignal=False):
		n = 0
		ct = 0.
		cd = 0.
		decided = False
		it = itertools.izip(targetSignal,distractorSignal)
		while not decided:
			try:
				t,d = it.next()
			except (GeneratorExit,StopIteration):
				break
			n+=1
			ct+=t
			cd+=d
			post_si_t = 1./(1./self.prior_si_t.**2+n/self.sigma_t**2)
			post_si_d = 1./(1./self.prior_si_d.**2+n/self.sigma_d**2)
			post_mu_t = (self.prior_mu_t/self.prior_si_t**2+ct/sigma_t**2)*post_si_t
			post_mu_d = (self.prior_mu_d/self.prior_si_d**2+cd/sigma_d**2)*post_si_d
			
			dif_mu = post_si_t-post_si_d
			dif_si = np.sqrt(post_si_t**2+post_si_d**2)
			if abs(dif_mu)/dif_si>=self.threshold:
				decided = True
			if storeSignal:
				self.signals.append([t,d])
		if decided:
			ret = decided,None,None,None,n
		else:
			performance = 1 if dif_mu>0 else 0
			ret = decided,performance,dif_mu,dif_si,n
		return ret
	
	def batchInference(self,targetSignalArr,distractorSignalArr):
		if any(targetSignalArr.shape!=distractorSignalArr):
			raise(ValueError("Target and distractor signal arrays must be numpy arrays with the same shape"))
		s = targetSignalArr.shape
		post_mu_t = self.prior_mu_t*np.ones(targetSignalArr.shape)
		post_mu_d = self.prior_mu_d*np.ones(targetSignalArr.shape)
		post_mu_t = self.prior_si_t*np.ones(targetSignalArr.shape)
		post_si_d = self.prior_si_d*np.ones(targetSignalArr.shape)
		n = np.tile(np.reshape(np.arange(s[1]),(1,s[1])),(s[0],1))
		
		post_si_t = 1./(1./self.prior_si_t.**2+n/self.sigma_t**2)
		post_si_d = 1./(1./self.prior_si_d.**2+n/self.sigma_d**2)
		post_mu_t = (self.prior_mu_t/self.prior_si_t**2+np.cumsum(targetSignalArr,axis=1)/sigma_t**2)*post_si_t
		post_mu_d = (self.prior_mu_d/self.prior_si_d**2+np.cumsum(distractorSignalArr,axis=1)/sigma_d**2)*post_si_d
		
		dif_mu = post_si_t-post_si_d
		dif_si = np.sqrt(post_si_t**2+post_si_d**2)
		ret = np.zeros(s[0],5)
		dec_time = np.argmax(np.abs(dif_mu)/dif_si>=self.threshold)
		for trial,dt in enumerate(dec_time):
			if dec_time==s[1]-1:
				if np.abs(dif_mu[trial,-1])/dif_si[trial,-1]<self.threshold:
					ret[trial] = False,None,None,None,s[1]
					continue
			

def posterior(targetmean,targetstd,distractormean,distractorstd):
	flucts = []
	post_mu_t = []
	post_mu_d = []
	post_si_t = []
	post_si_d = []
	for trial,(tm,ts,dm,ds) in enumerate(np.nditer([targetmean,targetstd,distractormean,distractorstd])):
		post_mu_t.append([prior_mu_t])
		post_mu_t.append([prior_mu_d])
		post_si_t.append([prior_si_t])
		post_si_t.append([prior_si_d])
		decided = False
		n = 0
		ct = 0.
		cd = 0.
		while not decided:
			target = random.gauss(tm,ts)
			distractor = random.gauss(tm,ts)
			ct+=target
			cd+=distractor
			n+=1
			post_si_t[-1].append(1./(1./prior_si_t.**2+n/sigma**2))
			post_si_d[-1].append(1./(1./prior_si_d.**2+n/sigma**2))
			post_mu_t[-1].append((prior_mu_t/prior_sigma_t**2+ct/sigma.^2).*post_si_t[-1][-1])
			post_mu_d[-1].append((prior_mu_d/prior_sigma_d**2+cd/sigma.^2).*post_si_d[-1][-1])
			
			dec_var = 
dprime = post_mu_t./post_sigma_t-post_mu_d./post_sigma_d;

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
    prob_acierto(j) = 1-normcdf(0,post_mu_t(s_tdec_ind(j))-post_mu_d(s_tdec_ind(j)),post_sigma_t(s_tdec_ind(j)),post_sigma_d(s_tdec_ind(j)));
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
            if dprime(ind)>0
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
