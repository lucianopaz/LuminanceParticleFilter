from __future__ import division

import enum, os, sys, math, scipy, pickle
import data_io as io
import cost_time as ct
import numpy as np
from utils import normpdf

class Location(enum.Enum):
	facu = 0
	home = 1
	cluster = 2
	unknown = 3

opsys,computer_name,kern,bla,bits = os.uname()
if opsys.lower().startswith("linux"):
	if computer_name=="facultad":
		loc = Location.facu
	elif computer_name.startswith("sge"):
		loc = Location.cluster
elif opsys.lower().startswith("darwin"):
	loc = Location.home
else:
	loc = Location.unknown

if loc==Location.facu:
	data_dir='/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles'
elif loc==Location.home:
	data_dir='/Users/luciano/Facultad/datos'
elif loc==Location.cluster:
	data_dir='/homedtic/lpaz/DecisionConfidenceKernels/data'
elif loc==Location.unknown:
	raise ValueError("Unknown data_dir location")
ISI = 0.04
distractor = 50.
patch_sigma = 5.
model_var = (patch_sigma**2)*2/ISI

def add_dead_time(gs,dt,dead_time_sigma,mode='full'):
	if dead_time_sigma==0.:
		return gs
	g1,g2 = gs
	conv_window = np.linspace(-dead_time_sigma*6,dead_time_sigma*6,int(math.ceil(dead_time_sigma*12/dt))+1)
	conv_window_size = conv_window.shape[0]
	conv_val = np.zeros_like(conv_window)
	conv_val[conv_window_size//2:] = normpdf(conv_window[conv_window_size//2:],0,dead_time_sigma)
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

def fit(subject,method="two_step"):
	dat,t,d = subject.load_data()
	mu,mu_indeces,count = np.unique((dat[:,0]-distractor)/ISI,return_inverse=True,return_counts=True)
	mus = np.concatenate((-mu[::-1],mu))
	counts = np.concatenate((count[::-1].astype(np.float64),count.astype(np.float64)))*0.5
	p = counts/np.sum(counts)
	
	prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	
	if method=="two_step":
		m = ct.DecisionPolicy(model_var=model_var,prior_mu_var=prior_mu_var,n=101,T=100,dt=ISI,reward=1,penalty=0,iti=1.,tp=0.,store_p=False)
		return scipy.optimize.fmin(two_step_merit,[0.],args=(m,dat,mu,mu_indeces),full_output=True)
	else:
		m = ct.DecisionPolicy(model_var=model_var,prior_mu_var=prior_mu_var,n=101,T=100,dt=ISI,reward=1,penalty=0,iti=1.,tp=0.,store_p=False)
		return scipy.optimize.fmin(full_merit,[0.,0.],args=(m,dat,mu,mu_indeces),full_output=True)

def dead_time_merit(dead_time_sigma,xub,xlb,m,dat,mu,mu_indeces):
	nlog_likelihood = 0.
	for index,drift in enumerate(mu):
		g1,g2 = add_dead_time(m.rt(drift,bounds=(xub,xlb)),m.dt,dead_time_sigma[0])
		for drift_trial in dat[mu_indeces==index]:
			rt = drift_trial[1]*1e-3 # Response times are stored in ms while simulations assume times are written in seconds
			perf = drift_trial[2]
			if perf==1:
				nlog_likelihood+=m.rt_nlog_like(g1,rt)
			else:
				nlog_likelihood+=m.rt_nlog_like(g2,rt)
	print nlog_likelihood
	return nlog_likelihood

def two_step_merit(cost,m,dat,mu,mu_indeces,output_list=None):
	m.cost = cost[0]
	xub,xlb = m.xbounds()
	dead_time_sigma,nlog_likelihood,niter,funcalls,warnflag = scipy.optimize.fmin(dead_time_merit,[0.],args=(xub,xlb,m,dat,mu,mu_indeces),full_output=True)
	if output_list is not None:
		del output_list[:]
		output_list.append(cost)
		output_list.append(dead_time_sigma)
	return nlog_likelihood

def full_merit(params,m,dat,mu,mu_indeces):
	cost = params[0]
	dead_time_sigma = params[1]
	
	nlog_likelihood = 0.
	m.cost = cost
	xub,xlb = m.xbounds()
	for index,drift in enumerate(mu):
		g1,g2 = add_dead_time(m.rt(drift,bounds=(xub,xlb)),m.dt,dead_time_sigma)
		for drift_trial in dat[mu_indeces==index]:
			rt = drift_trial[1]*1e-3 # Response times are stored in ms while simulations assume times are written in seconds
			perf = drift_trial[2]
			if perf==1:
				nlog_likelihood+=m.rt_nlog_like(g1,rt)
			else:
				nlog_likelihood+=m.rt_nlog_like(g2,rt)
	return nlog_likelihood

if __name__=="__main__":
	if len(sys.argv)>1:
		task = int(sys.argv[1])-1
		ntasks = int(sys.argv[2])
	else:
		task = 0
		ntasks = 1
	for i,s in enumerate(io.unique_subjects(data_dir)):
		if (i-task)%ntasks==0:
			pickle.dump(fit(s),"fit_subject_"+s.id,pickle.HIGHEST_PROTOCOL)
