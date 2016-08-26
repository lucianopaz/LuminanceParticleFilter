from __future__ import division

import pickle
import numpy as np
import data_io as io
import cost_time as ct
import moving_bounds_fits as mo
from matplotlib import pyplot as plt

f = open('fits/inference_fit_full_subject_1_seconds_iti_1-5.pkl','r')
out = pickle.load(f)
options = out['options']
fit_output = out['fit_output']
cost = fit_output[0]['cost']
dead_time = fit_output[0]['dead_time']
dead_time_sigma = fit_output[0]['dead_time_sigma']
phase_out_prob = fit_output[0]['phase_out_prob']
f.close()

f = open('fits/inference_fit_confidence_only_subject_1_seconds_iti_1-5.pkl','r')
out = pickle.load(f)
high_conf_thresh = out['fit_output'][0]['high_confidence_threshold']
f.close()
mo.set_time_units(options['time_units'])

# Load subject data
subjects = io.unique_subjects(mo.data_dir)
subject = subjects[0]
dat,t,d = subject.load_data()
if options['time_units']=='seconds':
	dat[:,1]*=1e-3

# Compute fitted theoretical distributions
ntrials = float(dat.shape[0])
max_RT = np.max(dat[:,1])
mu,mu_indeces,count = np.unique((dat[:,0]-mo.distractor)/mo.ISI,return_inverse=True,return_counts=True)
mu_prob = count.astype(np.float64)
mu_prob/=np.sum(mu_prob)
mus = np.concatenate((-mu[::-1],mu))
counts = np.concatenate((count[::-1].astype(np.float64),count.astype(np.float64)))*0.5
p = counts/np.sum(counts)
prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)

T = options['T']
dt = options['dt']
iti = options['iti']
tp = options['tp']
reward = options['reward']
penalty = options['penalty']
n = options['n']
if options['time_units']=='seconds':
	if T is None:
		T = 10.
	if iti is None:
		iti = 1.
	if tp is None:
		tp = 0.
else:
	if T is None:
		T = 10000.
	if iti is None:
		iti = 1000.
	if tp is None:
		tp = 0.
if dt is None:
	dt = mo.ISI

m1 = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=prior_mu_var,n=n,T=T,dt=dt,cost=cost,reward=reward,penalty=penalty,iti=iti,tp=tp,store_p=False)
m2 = ct.DecisionPolicy(model_var=mo.model_var,discrete_prior=(mu,mu_prob),n=n,T=T,dt=dt,cost=cost,reward=reward,penalty=penalty,iti=iti,tp=tp,store_p=False)

xb1 = np.array(m1.xbounds())
xb2 = np.array(m2.xbounds())
plt.figure()
plt.subplot(121)
plt.plot(m1.t,xb1.T,'--')
plt.plot(m1.t,xb2.T)
plt.subplot(122)
plt.plot(m1.t,m1.bounds.T,'--')
plt.plot(m1.t,m2.bounds.T)
