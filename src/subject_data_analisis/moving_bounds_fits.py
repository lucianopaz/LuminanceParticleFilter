from __future__ import division

import data_io as io
import cost_time as ct
import numpy as np
from utils import normpdf

def add_dead_time(gs,dt,dead_time_sigma,mode='full'):
	g1,g2 = gs
	conv_window = np.linspace(-dead_time_sigma*6,dead_time_sigma*6,int(math.ceil(dead_time_sigma*12/dt))+1)
	conv_window_size = conv_window.shape[0]
	conv_val = np.zeros_like(conv_window)
	conv_val[conv_window_size//2:] = np.normpdf(conv_window[conv_window_size//2:],0,dead_time_sigma)
	conv_val/=(np.sum(conv_val)*dt)
	cg1 = np.convolve(g1,conv_val,mode=mode)
	cg2 = np.convolve(g2,conv_val,mode=mode)
	a = int(0.5*(cg1.shape[0]-g1.shape[0]))
	if a==0:
		if cg1.shape[0]==g1.shape[0]:
			ret = (cg1,cg2)
		else:
			ret = (cg1[:-1],cg2[:-1])
	elif a==0.5*(cg1.shape[0]-g1.shape[0]):
		ret = (cg1[a:-a],cg2[a:-a])
	else:
		ret = (cg1[a:-a-1],cg2[a:-a-1])
	return ret

def rt_nlog_like(g1,rt):
	return 0.

if at_facu:
	data_dir='/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles'
elif at_casa:
	data_dir='/Users/luciano/Facultad/datos'
else:
	raise ValueError("Unknown data_dir location")

subjects = io.unique_subjects(data_dir)
ISI = 0.04
distractor = 50.
patch_sigma = 5.
model_var = (patch_sigma**2)*2/ISI
for s in subjects:
	dat,t,d = s.load_data()
	mu,mu_indeces,count = np.unique((dat[:,0]-distractor)/ISI,return_inverse=True,return_counts=True)
	mus = np.concatenate((-mu[::-1],mu))
	counts = np.concatenate((count[::-1].astype(np.float64),count.astype(np.float64)))*0.5
	p = counts/np.sum(counts)
	
	prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	
	# parameters cost and dead_time_sigma must be explored
	cost = 0.2
	dead_time_sigma = 0.2
	
	nlog_likelihood = 0.
	m = ct.DecisionPolicy(model_var=model_var,prior_mu_var=prior_mu_var,n=101,T=10,dt=ISI,reward=1,cost=cost,penalty=0,iti=1.,tp=0.,store_p=False)
	xub,xlb = m.xbounds()
	for index,drift in enumerate(mu):
		g1,g2 = add_dead_time(m.rt(drift,bounds=np.array([xub,xlb])),m.dt,dead_time_sigma)
		for drift_trial in dat[mu_indeces==index]:
			rt = drift_trial[1]
			perf = drift_trial[2]
			if perf==1:
				nlog_likelihood+=rt_nlog_like(g1,rt)
			else:
				nlog_likelihood+=rt_nlog_like(g2,rt)
