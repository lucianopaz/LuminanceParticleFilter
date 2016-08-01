from __future__ import division

import sys, pickle
import numpy as np
import matplotlib as mt
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import data_io as io
import cost_time as ct
import moving_bounds_fits as mo

subjects = io.unique_subjects(mo.data_dir)
files = ['fits/inference_fit_full_subject_'+str(sid)+'_seconds.pkl' for sid in range(1,7)]
files2 = ['fits/inference_fit_confidence_only_subject_'+str(sid)+'_seconds.pkl' for sid in range(1,7)]

subject = subjects[0]
fn = files[0]
fn2 = files2[0]

f = open(fn,'r')
out = pickle.load(f)
options = out['options']
fit_output = out['fit_output']
cost = fit_output[0]['cost']
dead_time = fit_output[0]['dead_time']
dead_time_sigma = fit_output[0]['dead_time_sigma']
phase_out_prob = fit_output[0]['phase_out_prob']
f.close()
f = open(fn2,'r')
out = pickle.load(f)
high_conf_thresh = out['fit_output'][0]['high_confidence_threshold']
f.close()
mo.set_time_units(options['time_units'])

# Load subject data
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

m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=prior_mu_var,n=n,T=T,dt=dt,reward=reward,penalty=penalty,iti=iti,tp=tp,store_p=False)
xub,xlb,v,v_explore,v1,v2 = m.xbounds(return_values=True)
m.invert_belief()
p = m.belief_transition_p()

mean_pg = np.sum(p*m.g,axis=2)
var_pg = np.sum(p*m.g**2,axis=2)-mean_pg**2

plt.figure()
plt.subplot(121)
plt.plot(m.t[:-1],var_pg/var_pg[0],'-',color='k')
plt.subplot(122)
plt.plot(m.t[:-1],np.mean(v_explore,axis=1))

#~ plt.figure()
#~ frameinterval = 0.01
#~ image = plt.imshow(p[0])
#~ cbar = plt.colorbar()
#~ plt.title('t = {0}'.format(m.t[0]))
#~ plt.pause(frameinterval)
#~ for i,pt in enumerate(p[1:]):
	#~ image.set_data(pt)
	#~ image.autoscale()
	#~ plt.title('t = {0}'.format(m.t[i+1]))
	#~ plt.pause(frameinterval)
#~ 
#~ plt.figure()
#~ #l, = plt.plot(m.g,np.diag(p[0]))
#~ l, = plt.plot(m.g,p[0][30])
#~ plt.title('t = {0}'.format(m.t[0]))
#~ plt.pause(frameinterval)
#~ for i,pt in enumerate(p[1:]):
	#~ #plt.plot(m.g,np.diag(pt))
	#~ l, = plt.plot(m.g,pt[30])
	#~ #l.set_data(m.g,np.diag(pt))
	#~ plt.title('t = {0}'.format(m.t[i+1]))
	#~ plt.pause(frameinterval)
