#!/usr/bin/python
#-*- coding: UTF-8 -*-

from __future__ import division
import numpy as np
import data_io as io
import kernels as ke
import perfect_inference as pe
try:
	import matplotlib as mt
	from matplotlib import pyplot as plt
	interactive_pyplot = plt.isinteractive()
	loaded_plot_libs = True
except:
	loaded_plot_libs = False
import cma, os, math, pickle, sys

_vectErf = np.vectorize(math.erf,otypes=[np.float])
def normcdf(x,mu=0.,sigma=1.):
	"""
	Compute normal cummulative distribution with mean mu and standard
	deviation sigma. x can be a numpy array.
	"""
	try:
		new_x = (x-mu)/sigma
	except ZeroDivisionError:
		new_x = np.sign(x-mu)*np.inf
	return 0.5 + 0.5*_vectErf(new_x / np.sqrt(2.0))

def objective(params,model,performance,rt_ind,target,distractor,return_ind=False,return_likelihood=False,return_performance=False):
	model.var_t = params[0]**2
	model.var_d = params[0]**2
	if len(params)>1:
		model.prior_va_t = params[1]**2
		model.prior_va_d = params[1]**2
		if len(params)>2:
			model.prior_mu_t = params[2]
			model.prior_mu_d = params[2]
	mu_t,mu_d,va_t,va_d = model.passiveInference(target,distractor)
	mu_diff = mu_t[:,0:]-mu_d[:,0:] # We disregard the prior value that is stored in [:,0]
	va_diff = va_t[:,0:]+va_d[:,0:] # We disregard the prior value that is stored in [:,0]
	miss_prob = ke.center_around(normcdf(0.,mu_diff,va_diff),rt_ind)
	likelihood = miss_prob.copy()
	likelihood[performance==1] = 1.-likelihood[performance==1]
	likelihood = np.nanmean(np.log(likelihood),axis=0)*target.shape[0]
	ind = np.nanargmax(likelihood[:mu_t.shape[1]])
	out = -likelihood[ind] # The minus is added so the that when fmin is called, the maximum is found
	if any([return_ind,return_likelihood,return_performance]):
		out = (out,)
		if return_ind:
			out+=(ind,)
		if return_likelihood:
			out+=(likelihood,)
		if return_performance:
			performance = np.nanmean(1-miss_prob,axis=0)
			performance_std = np.sqrt(np.nansum((1-miss_prob)*miss_prob,axis=0))/np.sum(~np.isnan(miss_prob),axis=0)
			out+=([performance,performance_std],)
	return out

def infere_variance(subject,x0,lbounds=None,ubounds=None,cma_sigma=1/3,cma_options=cma.CMAOptions(),restarts=0):
	if not isinstance(cma_options,cma.CMAOptions):
		cma_options = cma.CMAOptions(cma_options)
	if lbounds is None:
		lbounds = 5*np.ones(len(x0))
	else:
		lbounds = np.array(lbounds)
	if ubounds is None:
		ubounds = np.inf*np.ones(len(x0))
	else:
		ubounds = np.array(ubounds)
	if any(np.isinf(ubounds)):
		scaling_of_variables = None
	else:
		scaling_of_variables = ubounds-lbounds
	cma_options.set('bounds',[lbounds,ubounds])
	if scaling_of_variables is not None:
		cma_options.set('scaling_of_variables',scaling_of_variables)
	
	dat,t,d = subject.load_data()
	inds = dat[:,1]<1000
	dat = dat[inds]
	t = np.mean(t[inds],axis=2)
	d = np.mean(d[inds],axis=2)
	rt_ind = np.floor(dat[:,1]/40)
	model = pe.KnownVarPerfectInference(prior_mu_t=50,prior_mu_d=50,prior_va_t=15**2,prior_va_d=15**2)
	args = (model,dat[:,2],rt_ind,t,d)
	
	cma_fmin_output = cma.fmin(objective,x0,cma_sigma,options=cma_options,args=args,restarts=restarts)
	stop_cond = {}
	for key in cma_fmin_output[7].keys():
		stop_cond[key] = cma_fmin_output[7][key]
	return {'xopt':cma_fmin_output[0],'fopt':cma_fmin_output[1],'evalsopt':cma_fmin_output[2],\
			'evals':cma_fmin_output[3],'iterations':cma_fmin_output[4],'xmean':cma_fmin_output[5],\
			'stds':cma_fmin_output[6],'stop':stop_cond}

def fit(data_dir='/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles'):
	# Fit parameters for each subject and then for all subjects
	subjects = io.unique_subjects(data_dir)
	subjects.append(io.merge_subjects(subjects))
	x0 = np.array([5,15])
	counter = 0
	for s in subjects:
		counter+=1
		print counter
		output = infere_variance(s,x0,lbounds=[5,5],ubounds=[50,50])
		if len(x0)==1:
			fittype = 'var'
		elif len(x0)==2:
			fittype = 'var-pva'
		elif len(x0)==3:
			fittype = 'var-pva-pmu'
		filename = 'fits/Infere_subject_'+str(s.id)+'_type_'+fittype+'.pkl'
		f = open(filename,'wb')
		pickle.dump((s,output),f,pickle.HIGHEST_PROTOCOL)
		f.close()

def analyze(fit_dir='fits/',save=False,savefname='inferences'):
	if not fit_dir.endswith('/'):
		fit_dir+='/'
	files = sorted([f for f in os.listdir(fit_dir) if (f.endswith(".pkl") and f.startswith('Infere_'))])
	figs = []
	for f in files:
		ff = open(fit_dir+f,'r')
		(subject,fit_output) = pickle.load(ff)
		ff.close()
		if subject.id<=1:
			print subject.name, subject.id
			print subject.data_files
		dat,t,d = subject.load_data()
		inds = dat[:,1]<1000
		dat = dat[inds]
		t = np.mean(t[inds],axis=2)
		d = np.mean(d[inds],axis=2)
		rt_ind = np.floor(dat[:,1]/40)
		fluctuations = np.transpose(np.array([t.T-dat[:,0],d.T-50]),(2,0,1))
		rdk,rck,rdk_std,rck_std = ke.kernels(fluctuations,1-dat[:,2],dat[:,3]-1,locked_on_onset=False,RT_ind=rt_ind)
		centered_ind = np.arange(rdk.shape[1],dtype=float)
		rT = (centered_ind-t.shape[1]+1)*0.04
		
		model = pe.KnownVarPerfectInference()
		max_logp,ind_max,likelihood,perf = objective(fit_output['xopt'],model,dat[:,2],rt_ind,t,d,return_ind=True,return_likelihood=True,return_performance=True)
		performance = perf[0]
		performance_std = perf[1]
		
		if not interactive_pyplot:
			plt.ioff() # cma overrides original pyplot settings and if interactive mode was off, the plot is never displayed!

		figs.append(plt.figure(figsize=(13,10)))
		ax = plt.subplot(221)
		plt.plot(rT,rdk[0],color='b',label='$D_{S}$')
		plt.plot(rT,rdk[1],color='r',label='$D_{N}$')
		plt.fill_between(rT,rdk[0]-rdk_std[0],rdk[0]+rdk_std[0],color='b',alpha=0.3,edgecolor=None)
		plt.fill_between(rT,rdk[1]-rdk_std[1],rdk[1]+rdk_std[1],color='r',alpha=0.3,edgecolor=None)
		plt.plot(rT[ind_max]*np.ones(2),ax.get_ylim(),'--k')
		plt.ylabel('Decision kernel')
		
		plt.subplot(223)
		plt.plot([-1,1],np.mean(dat[:,2])*np.ones(2),'--k',label='Subject')
		plt.errorbar(np.linspace(-1,1,performance.shape[0]), performance, yerr=performance_std,label="$\sigma=%1.2f$"%(fit_output['xopt'][0]))
		plt.legend(loc=0)
		plt.xlabel('T locked on RT [s]')
		plt.ylabel('Performance')
		
		ax = plt.subplot(122)
		plt.plot(np.linspace(-1,1,likelihood.shape[0]),likelihood)
		ylim = ax.get_ylim()
		plt.plot(np.linspace(-1,1,likelihood.shape[0])[ind_max]*np.ones(2),ylim,'--k')
		ax.set_ylim(ylim)
		plt.xlabel('T locked on RT [s]')
		plt.ylabel('Log likelihood')
		
		sname = str(subject.id) if subject.id!=0 else 'all'
		plt.suptitle('Subject '+sname)
		
		#~ figs[-1].tight_layout()
	if save:
		from matplotlib.backends.backend_pdf import PdfPages
		with PdfPages('../../figs/'+savefname+'.pdf') as pdf:
			for fig in figs:
				pdf.savefig(fig)
	else:
		plt.show()

if __name__=="__main__":
	if len(sys.argv)>1:
		if sys.argv[1]=='0' or sys.argv[1].lower()=='fit':
			if len(sys.argv)>2:
				fit(sys.argv[2])
			else:
				fit()
		elif sys.argv[1]=='1' or sys.argv[1].lower()=='analyze':
			if len(sys.argv)>2:
				analyze(save=sys.argv[2].lower()=='true')
			else:
				analyze()
	else:
		fit()
