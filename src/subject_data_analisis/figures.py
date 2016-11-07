#-*- coding: UTF-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

import sys, pickle
import numpy as np
import matplotlib as mt
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
import analysis, utils
import data_io as io
import data_io_cognition as io_cog
import cost_time as ct
import moving_bounds_fits as mo
import fits_cognition as fits
from matplotlib import image as mpimg
from matplotlib.colors import LogNorm
from fits_cognition import Fitter_plot_handler

def place_axes_subfig_label(ax,label,horizontal=-0.05,vertical=1.05,verticalalignment='top',horizontalalignment='right',**kwargs):
	if isinstance(ax,Axes3D):
		ax.text2D(horizontal, vertical, label,transform=ax.transAxes,
				verticalalignment=verticalalignment, horizontalalignment=horizontalalignment,
				**kwargs)
	else:
		ax.text(horizontal, vertical, label,transform=ax.transAxes,
				verticalalignment=verticalalignment, horizontalalignment=horizontalalignment,
				**kwargs)

def grouped_bar_plot(values,ax=None,ebars=None,annotations=None,colors=None,
					group_width=0.9,group_member_names=None,group_names=None,
					barkwargs={},ebarkwargs={},annotkwargs={},colormap='rainbow'):
	nmembers,ngroups = values.shape
	bar_width = group_width/nmembers
	bar_pos = np.arange(ngroups)
	group_center = bar_pos+0.5*group_width
	if colors is None:
		if isinstance(colormap,str):
			colormap = plt.get_cmap(colormap)
		colors = [colormap(x) for x in np.linspace(0,1,ngroups)]
	all_positive_values = np.all(values>=0)
	all_bars = []
	all_ebars = []
	all_texts = []
	if ax is None:
		ax = plt.gca()
	for i,vals in enumerate(values):
		all_bars.append(ax.bar(bar_pos+i*bar_width,vals,bar_width,color=colors[i],**barkwargs))
		heights = vals
		if not ebars is None:
			all_ebars.append(ax.errorbar(bar_pos+(i+0.5)*bar_width,vals,ebars[i],color=colors[i],marker='',linestyle='',**ebarkwargs))
			heights+= (ebars[i]*np.sign(vals))
		if not annotations is None:
			texts = []
			for j,ann in enumerate(annotations[i]):
				texts.append(ax.text(bar_pos[j]+(i+0.5)*bar_width,heights[j],ann, ha='center', va='bottom' if heights[j]>=0 else 'top',**annotkwargs))
			all_texts.append(texts)
	ax.set_xticks(group_center)
	if group_names:
		ax.set_xticklabels(group_names)
	else:
		ax.set_xticklabels(np.arange(ngroups)+1)
	if group_member_names:
		ax.legend([r[0] for r in all_bars],group_member_names,loc='best', fancybox=True, framealpha=0.5)
	if not all_positive_values:
		ax.plot(ax.get_xlim(),[0,0],'k')
	else:
		ylim = ax.get_ylim()
		ax.set_ylim([0,ylim[1]])
	
	return all_bars,all_ebars,all_texts

def value_and_bounds_sketch(fname='value_and_bounds_sketch',suffix='.svg'):
	fname+=suffix
	mo.set_time_units('seconds')
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=4990.24,n=101,T=30,dt=0.01,cost=0.1,reward=1,penalty=0,iti=1.5,tp=0.)
	m.xbounds()
	m.refine_value(n=501)
	xub,xlb,value,v_explore,v1,v2 = m.xbounds_fixed_rho(set_bounds=True,return_values=True)
	xb = np.array([xub,xlb])
	b = m.bounds

	plt.figure(figsize=(11,8))
	# Plot value matrix
	plt.subplot(221)
	plt.imshow(value[m.t<3].T,aspect='auto',cmap='jet',interpolation='none',origin='lower',extent=[m.t[0],3,m.g[0],m.g[-1]])
	cbar = plt.colorbar(orientation='vertical')
	cbar.set_label(r'$\tilde V$')
	plt.ylabel(r'$g$')
	plt.xlabel('T [s]')
	
	# Plot how the value of exploring changes through time
	plt.subplot(223)
	plt.plot(m.g,np.max(np.array([v1,v2]),axis=0),linestyle='--',color='k',linewidth=3,zorder=2.1)
	#~ inds = np.array([0,50,100,250,len(m.t)-2])
	inds = np.array([0,10,20,30,50])
	cmap = plt.get_cmap('RdYlGn')
	#~ for ind,cmap_ind in zip(inds,1-inds.astype(np.float64)/float((len(m.t)-1)*10./m.T)):
	for ind,cmap_ind in zip(inds,1-inds.astype(np.float64)/float((np.max(inds)))):
		plt.plot(m.g,v_explore[ind],color=cmap(cmap_ind),linewidth=2)
	ylim = list(plt.gca().get_ylim())
	ylim[0] = -0.8
	plt.gca().set_ylim(tuple(ylim))
	g_area = np.logical_and(m.g<m.bounds[0][inds[-1]],m.g>m.bounds[1][inds[-1]])
	gs = m.g[g_area]
	top_region = np.max(np.array([v1,v2]),axis=0)[g_area]
	low_region = np.ones_like(top_region)*plt.gca().get_ylim()[0]
	plt.fill_between(gs,low_region,top_region, interpolate=True,facecolor='gray',zorder=5,edgecolor=None)
	plt.xlabel(r'$g$')
	plt.ylabel(r'$\tilde V$')
	
	# Plot decision bounds in belief space
	plt.subplot(222)
	plt.plot(m.t[m.t<3],b[0,m.t<3],linewidth=2,color='b')
	plt.plot(m.t[m.t<3],b[1,m.t<3],linewidth=2,color='r')
	plt.ylabel('$g$ bound')
	plt.gca().set_ylim([0,1])
	# Plot decision bounds in the diffusion space
	plt.subplot(224)
	plt.plot(m.t[m.t<3],xb[0,m.t<3].T,linewidth=2,color='b')
	plt.plot(m.t[m.t<3],xb[1,m.t<3].T,linewidth=2,color='r')
	
	samples = ct.diffusion_path_samples(0,mo.model_var,m.dt,m.T,xb)
	ndec = [2,2]
	for sample in samples:
		if sample['dec']:
			if ndec[sample['dec']-1]>0:
				ndec[sample['dec']-1]-=1
				y = sample['x']
				if sample['dec']==1:
					color = 'b'
					y[-1] = xub[len(y)-1]
				else:
					color = 'r'
					y[-1] = xlb[len(y)-1]
				plt.plot(sample['t'],sample['x'],color=color)
	
	plt.ylabel(r'$x(t)$ bound')
	plt.xlabel('T [s]')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def bounds_vs_cost(fname='bounds_cost',suffix='.svg',n_costs=20,maxcost=1.,prior_mu_var=4990.24,n=101,T=10.,dt=None,reward=1,penalty=0,iti=1.5,tp=0.):
	fname+=suffix
	mo.set_time_units('seconds')
	if dt is None:
		dt = mo.ISI
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=prior_mu_var,n=n,T=T,dt=dt,reward=reward,penalty=penalty,iti=iti,tp=tp)
	costs = np.linspace(0,maxcost,n_costs)
	s1_colors = [plt.get_cmap('YlGn')(x) for x in np.linspace(1,0,n_costs)]
	s2_colors = [plt.get_cmap('YlOrRd')(x) for x in np.linspace(1,0,n_costs)]
	
	plt.figure(figsize=(8,4))
	gs1 = gridspec.GridSpec(1, 1)
	gs1.update(left=0.1, right=0.80)
	gs2 = gridspec.GridSpec(1, 1)
	gs2.update(left=0.85, right=0.9)
	ax = plt.subplot(gs1[0])
	plt.sca(ax)
	for cost,c1,c2 in zip(costs,s1_colors,s2_colors):
		m.set_cost(cost)
		xub,xlb = m.xbounds()
		plt.plot(m.t,xub,color=c1)
		plt.plot(m.t,xlb,color=c2)
	plt.ylabel('$x(t)$ Bounds',fontsize=16)
	plt.xlabel('T [s]',fontsize=16)
	
	cbar_ax = plt.subplot(gs2[0])
	cbar = np.array([[plt.get_cmap('YlGn')(x) for x in np.linspace(1,0,100)],
					 [plt.get_cmap('YlOrRd')(x) for x in np.linspace(1,0,100)]])
	cbar = np.swapaxes(cbar,0,1)
	plt.sca(cbar_ax)
	cbar_ax.yaxis.set_label_position("right")
	cbar_ax.tick_params(reset=True,which='major',axis='y',direction='in',left=True, right=True,bottom=False,top=False,labelleft=False, labelright=True)
	plt.imshow(cbar,aspect='auto',cmap=None,interpolation='none',origin='lower',extent=[0,1,0,maxcost])
	cbar_ax.xaxis.set_ticks([])
	
	plt.ylabel('Cost [Hz]',fontsize=16)
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def rt_fit(fname='rt_fit',suffix='.svg'):
	fname+=suffix
	subjects = io.unique_subjects(mo.data_dir)
	files = ['fits/inference_fit_full_subject_'+str(sid)+'_seconds_iti_1-5.pkl' for sid in range(1,7)]
	files2 = ['fits/inference_fit_confidence_only_subject_'+str(sid)+'_seconds_iti_1-5.pkl' for sid in range(1,7)]
	alldata = None
	all_sim_rt = None
	for subject,fn,fn2 in zip(subjects,files,files2):
		# Load fit parameters
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
		if alldata is None:
			alldata = dat
		else:
			alldata = np.vstack((alldata,dat))
		
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
		
		m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=prior_mu_var,n=n,T=T,dt=dt,cost=cost,reward=reward,penalty=penalty,iti=iti,tp=tp,store_p=False)
		sim_rt = mo.theoretical_rt_distribution(cost,dead_time,dead_time_sigma,phase_out_prob,m,mu,mu_prob,max_RT,[high_conf_thresh])
		if all_sim_rt is None:
			all_sim_rt = {'hit':sim_rt['full']['all'][0]*ntrials,
						  'miss':sim_rt['full']['all'][1]*ntrials,
						  'high':np.sum(sim_rt['full']['high'],axis=0)*ntrials,
						  'low':np.sum(sim_rt['full']['low'],axis=0)*ntrials}
		else:
			all_sim_rt['hit']+= sim_rt['full']['all'][0]*ntrials
			all_sim_rt['miss']+= sim_rt['full']['all'][1]*ntrials
			all_sim_rt['high']+= np.sum(sim_rt['full']['high'],axis=0)*ntrials
			all_sim_rt['low']+= np.sum(sim_rt['full']['low'],axis=0)*ntrials
	# Normalize by the number of fitted trials
	ntrials = float(alldata.shape[0])
	all_sim_rt['hit']/=ntrials
	all_sim_rt['miss']/=ntrials
	all_sim_rt['high']/=ntrials
	all_sim_rt['low']/=ntrials
	
	# Compute subject histograms
	rt = alldata[:,1]
	perf = alldata[:,2]
	conf = alldata[:,3]
	
	temp,edges = np.histogram(rt,100)
	
	high_hit_rt,temp = np.histogram(rt[np.logical_and(perf==1,conf==2)],edges)
	low_hit_rt,temp = np.histogram(rt[np.logical_and(perf==1,conf==1)],edges)
	high_miss_rt,temp = np.histogram(rt[np.logical_and(perf==0,conf==2)],edges)
	low_miss_rt,temp = np.histogram(rt[np.logical_and(perf==0,conf==1)],edges)
	
	high_hit_rt = high_hit_rt.astype(np.float64)
	low_hit_rt = low_hit_rt.astype(np.float64)
	high_miss_rt = high_miss_rt.astype(np.float64)
	low_miss_rt = low_miss_rt.astype(np.float64)
	
	hit_rt = high_hit_rt+low_hit_rt
	miss_rt = high_miss_rt+low_miss_rt
	
	xh = np.array([0.5*(x+y) for x,y in zip(edges[1:],edges[:-1])])
	
	normalization = np.sum(hit_rt+miss_rt)*(xh[1]-xh[0])
	hit_rt/=normalization
	miss_rt/=normalization
	
	high_hit_rt/=normalization
	high_miss_rt/=normalization
	low_hit_rt/=normalization
	low_miss_rt/=normalization
	
	# Plot decision data
	mxlim = np.ceil(np.max(rt))
	plt.figure(figsize=(10,5))
	ax1 = plt.subplot(121)
	plt.step(xh,hit_rt,label='Subjects hit',where='post',color='b')
	plt.step(xh,-miss_rt,label='Subjects miss',where='post',color='r')
	plt.plot(m.t,all_sim_rt['hit'],label='Theoretical hit',linewidth=3,color='b')
	plt.plot(m.t,-all_sim_rt['miss'],label='Theoretical miss',linewidth=3,color='r')
	plt.xlim([0,mxlim])
	if options['time_units']=='seconds':
		plt.xlabel('T [s]')
	else:
		plt.xlabel('T [ms]')
	plt.ylabel('Prob density')
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	# Plot confidence data
	ax2 = plt.subplot(122,sharey=ax1)
	plt.step(xh,high_hit_rt+high_miss_rt,label='Subjects high',where='post',color='forestgreen')
	plt.step(xh,-(low_hit_rt+low_miss_rt),label='Subjects low',where='post',color='mediumpurple')
	plt.plot(m.t,all_sim_rt['high'],label='Theoretical high',linewidth=3,color='forestgreen')
	plt.plot(m.t,-all_sim_rt['low'],label='Theoretical low',linewidth=3,color='mediumpurple')
	plt.xlim([0,mxlim])
	plt.gca().tick_params(labelleft=False)
	if options['time_units']=='seconds':
		plt.xlabel('T [s]')
	else:
		plt.xlabel('T [ms]')
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	
	place_axes_subfig_label(ax1,'A',horizontal=-0.12,vertical=1.02,fontsize='24')
	place_axes_subfig_label(ax2,'B',horizontal=-0.05,vertical=1.02,fontsize='24')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def prior_sketch(fname='prior_sketch',suffix='.svg'):
	fname+=suffix
	from scipy import stats
	mo.set_time_units('seconds')
	subjects = io.unique_subjects(mo.data_dir)
	subject = io.merge_subjects(subjects)
	dat,t,d = subject.load_data()
	mu,mu_indeces,count = np.unique((dat[:,0]-mo.distractor)/mo.ISI,return_inverse=True,return_counts=True)
	mus = np.concatenate((-mu[::-1],mu))
	counts = np.concatenate((count[::-1].astype(np.float64),count.astype(np.float64)))*0.5
	p = counts/np.sum(counts)
	
	prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	
	plt.figure(figsize=(8,4))
	dmu = mus[1]-mus[0]
	plt.bar(mus-dmu*0.4,p/dmu,width=dmu*0.8,color='sage',label=r'$\mu$ distribution')
	x = np.linspace(mus[0],mus[-1],1000)
	plt.plot(x,stats.norm.pdf(x,0,np.sqrt(prior_mu_var)),linewidth=2,color='darkred',label='Conjugate prior')
	plt.xlabel(r'$\mu \left[\frac{\mathrm{cd}}{\mathrm{m}^{2}\mathrm{s}}\right]$',fontsize=22)
	plt.ylabel('Prob density',fontsize=16)
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def confidence_sketch(fname='confidence_sketch',suffix='.svg'):
	fname+=suffix
	mo.set_time_units('seconds')
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=4990.24,n=101,T=10.,dt=mo.ISI,cost=0.1,reward=1.,penalty=0.,iti=1.5,tp=0.,store_p=False)
	dead_time = 0.3
	dead_time_sigma = 0.4
	phase_out_prob = 0.05
	xub,xlb = m.xbounds()
	log_odds = m.log_odds()
	high_conf_threshold = 0.5
	
	_dt = 0.001
	_nT = int(m.T/_dt)+1
	_t = np.arange(_nT)*_dt
	xub = np.interp(_t,m.t,xub)
	xlb = np.interp(_t,m.t,xlb)
	log_odds = np.array([np.interp(_t,m.t,log_odds[0]),np.interp(_t,m.t,log_odds[1])])
	
	ind_bound = (log_odds[0]<=high_conf_threshold).nonzero()[0][0]
	
	
	plt.figure(figsize=(8,6))
	ax1=plt.subplot(211)
	p0, = plt.plot(_t,log_odds[0],color='k',linewidth=3,label=r'$C(t)$')
	plt.plot([0,3],high_conf_threshold*np.ones(2),'--k')
	p1 = plt.Rectangle((0, 0), 1, 1, fc="forestgreen")
	p2 = plt.Rectangle((0, 0), 1, 1, fc="mediumpurple")
	plt.fill_between(_t[:ind_bound+1],log_odds[0][:ind_bound+1],interpolate=True,color='forestgreen',alpha=0.6)
	plt.fill_between(_t[ind_bound:],log_odds[0][ind_bound:],interpolate=True,color='mediumpurple',alpha=0.6)
	plt.legend([p0,p1, p2], [r'$C(t)$','High confidence zone', 'Low confidence zone'],loc='best', fancybox=True, framealpha=0.5)
	plt.ylabel('Log odds')
	ax1.tick_params(labelleft=True,labelbottom=False)
	
	ax2 = plt.subplot(212,sharex=ax1)
	plt.plot(_t,xub,color='b')
	plt.plot(_t,xlb,color='r')
	plt.fill_between(_t[:ind_bound+1],xub[:ind_bound+1],xlb[:ind_bound+1],interpolate=True,color='forestgreen',alpha=0.6)
	plt.fill_between(_t[ind_bound:],xub[ind_bound:],xlb[ind_bound:],interpolate=True,color='mediumpurple',alpha=0.6)
	ax1.set_xlim([0,3])
	plt.ylabel(r'Bound in $x(t)$ space')
	plt.xlabel('T [s]')
	
	place_axes_subfig_label(ax1,'A',horizontal=-0.08,vertical=1.09,fontsize='24')
	place_axes_subfig_label(ax2,'B',horizontal=-0.09,vertical=1.09,fontsize='24')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def decision_rt_sketch(fname='decision_rt_sketch',suffix='.svg'):
	fname+=suffix
	subjects = io.unique_subjects(mo.data_dir)
	subject = io.merge_subjects(subjects)
	dat,t,d = subject.load_data()
	mu_data = (dat[:,0]-mo.distractor)/mo.ISI
	mu,mu_indeces,count = np.unique(mu_data,return_inverse=True,return_counts=True)
	mu_prob = count.astype(np.float64)/np.sum(count.astype(np.float64))
	if mu[0]==0:
		mus = np.concatenate((-mu[-1:0:-1],mu))
		p = np.concatenate((mu_prob[-1:0:-1],mu_prob))
		p[mus!=0]*=0.5
	else:
		mus = np.concatenate((-mu[::-1],mu))
		p = np.concatenate((mu_prob[::-1],mu_prob))*0.5
	
	prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
	
	mo.set_time_units('seconds')
	
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=prior_mu_var,n=101,T=10.,dt=mo.ISI,cost=0.1,reward=1.,penalty=0.,iti=1.5,tp=0.,store_p=False)
	dead_time = 0.3
	dead_time_sigma = 0.4
	phase_out_prob = 0.05
	xub,xlb = m.xbounds()
	
	difusion_teo_rt = None
	model_teo_rt = None
	
	max_RT = np.max(dat[:,1]*1e-3)
	_max_RT = m.t[np.ceil(max_RT/m.dt)]
	phased_out_rt = np.zeros_like(m.t)
	phased_out_rt[m.t<_max_RT] = 1./(_max_RT)
	for drift,drift_prob in zip(mu,mu_prob):
		g = np.array(m.rt(drift,bounds=(xub,xlb)))
		if difusion_teo_rt is None:
			difusion_teo_rt = g[:]*drift_prob
		else:
			difusion_teo_rt+= g[:]*drift_prob
		g1,g2 = mo.add_dead_time(g,m.dt,dead_time,dead_time_sigma)
		g1 = g1*(1-phase_out_prob)+0.5*phase_out_prob*phased_out_rt
		g2 = g2*(1-phase_out_prob)+0.5*phase_out_prob*phased_out_rt
		if model_teo_rt is None:
			model_teo_rt = np.array([g1,g2])*drift_prob
		else:
			model_teo_rt+= np.array([g1,g2])*drift_prob
	
	xb = np.array([xub,xlb])
	log_odds = m.log_odds()
	difusion_sim_rt,difusion_sim_dec = ct.sim_rt(mu_data,mo.model_var,m.dt,m.T,xb)
	model_sim_rt,model_sim_dec = mo.sim_rt(mu_data,mo.model_var,m.dt,m.T,max_RT,xb,log_odds,dead_time,dead_time_sigma,phase_out_prob,cost_time_sim_rt_output=(difusion_sim_rt,difusion_sim_dec))
	
	edges = m.t[:np.ceil(max_RT/m.dt)+1]
	centers = [0.5*(em+eM) for em,eM in zip(edges[:-1],edges[1:])]
	
	difusion_sim_hit_hist,edges = np.histogram(difusion_sim_rt[difusion_sim_dec==1],edges)
	difusion_sim_hit_hist = difusion_sim_hit_hist.astype(np.float64)
	difusion_sim_miss_hist,edges = np.histogram(difusion_sim_rt[difusion_sim_dec==2],edges)
	difusion_sim_miss_hist = difusion_sim_miss_hist.astype(np.float64)
	norm = np.sum(difusion_sim_hit_hist+difusion_sim_miss_hist)*(edges[1]-edges[0])
	difusion_sim_hit_hist/=norm
	difusion_sim_miss_hist/=norm
	
	likeliest_mu = [mu for mu,prob in zip(mu,mu_prob) if prob==np.max(mu_prob)][0]
	dense_t = np.arange(0,int(m.T*1e3)+1)*1e-3
	dense_bounds = np.array([np.interp(dense_t, m.t, xub),np.interp(dense_t, m.t, xlb)])
	samples = ct.diffusion_path_samples(likeliest_mu,mo.model_var,1e-3,m.T,dense_bounds,reps=50)
	
	model_sim_hit_hist,edges = np.histogram(model_sim_rt[model_sim_dec==1],edges)
	model_sim_hit_hist = model_sim_hit_hist.astype(np.float64)
	model_sim_miss_hist,edges = np.histogram(model_sim_rt[model_sim_dec==2],edges)
	model_sim_miss_hist = model_sim_miss_hist.astype(np.float64)
	norm = np.sum(model_sim_hit_hist+model_sim_miss_hist)*(edges[1]-edges[0])
	model_sim_hit_hist/=norm
	model_sim_miss_hist/=norm
	
	plt.figure(figsize=(12,6))
	gs1 = gridspec.GridSpec(3, 1, left=0.05, right=0.32,hspace=0, height_ratios=[2,1,1])
	gs2 = gridspec.GridSpec(1, 1, left=0.38, right=0.51, bottom=0.27, top=0.52)
	gs3 = gridspec.GridSpec(1, 1, left=0.65, right=0.95)
	ax1 = plt.subplot(gs1[0,0])
	plt.step(edges[:-1],difusion_sim_hit_hist,where='pre',color='b',linewidth=1,label='Simulation')
	plt.plot(m.t,difusion_teo_rt[0],color='b',linewidth=3,label='Theoretical')
	plt.ylabel('Up RT dist')
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.get_xaxis().tick_bottom()
	ax1.get_yaxis().tick_left()
	ax1.tick_params(labelleft=True,labelbottom=False)
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	plt.title('Diffusion first\npassage time')
	
	ax3 = plt.subplot(gs1[2,0],sharex=ax1)
	plt.step(edges[:-1],difusion_sim_miss_hist,where='pre',color='r',linewidth=1)
	plt.plot(m.t,difusion_teo_rt[1],color='r',linewidth=3)
	ax3.set_ylim(np.array(ax1.get_ylim()[::-1])*0.5)
	ax3.spines['right'].set_visible(False)
	ax3.get_xaxis().tick_bottom()
	ax3.get_yaxis().tick_left()
	ax3.tick_params(labelleft=True)
	plt.ylabel('Down RT dist')
	plt.xlabel('T [s]')
	
	max_b = np.max(xb)
	ax2 = plt.subplot(gs1[1,0],sharex=ax1)
	ax2.spines['right'].set_visible(False)
	plt.plot(m.t,xub,color='b',linestyle='--')
	plt.plot(m.t,xlb,color='r',linestyle='--')
	ax2.set_yticks([])
	ax2.tick_params(labelbottom=False)
	ax2.set_ylim([-np.ceil(max_b/10)*10,np.ceil(max_b/10)*10])
	
	ndec = [1,1]
	ax2 = plt.subplot(gs1[1,0],sharex=ax1)
	for sample in reversed(samples):
		if sample['dec'] and sample['rt']>=0.15:
			if ndec[sample['dec']-1]>0:
				ndec[sample['dec']-1]-=1
				y = sample['x']
				if sample['dec']==1:
					color = 'b'
					y[-1] = dense_bounds[0][len(y)-1]
				else:
					color = 'r'
					y[-1] = dense_bounds[1][len(y)-1]
				plt.plot(sample['t'],sample['x'],color=color)
	plt.ylabel('$x(t)$',fontsize=15)
	plt.gca().set_xlim([0,1])
	
	ax1.set_zorder(0.2)
	ax2.set_zorder(0)
	ax3.set_zorder(0.2)
	
	ax_conv = plt.subplot(gs2[0])
	from scipy import stats
	conv_x = np.linspace(0,2)
	conv_y = np.zeros_like(conv_x)
	conv_y[conv_x>=dead_time] = stats.norm.pdf(conv_x[conv_x>=dead_time],dead_time,dead_time_sigma)
	plt.plot(conv_x,conv_y)
	ax_conv.set_xlim([0,2])
	ax_conv.set_yticks([])
	ax_conv.set_xticks([])
	plt.title('Convolve with\nno decision time')
	plt.xlabel('T')
	ax1.annotate('', xy=[0.38,0.40], xytext=[0.25,0.25],
					size=10, va="center", ha="center",
					xycoords=('figure fraction'), textcoords=(ax1,'axes fraction'),
					arrowprops=dict(arrowstyle='simple',connectionstyle='arc3,rad=-0.3',color='gray'))
	ax3.annotate('', xy=[0.38,0.39], xytext=[0.25,0.75],
					size=10, va="center", ha="center",
					xycoords=('figure fraction'), textcoords=(ax3,'axes fraction'),
					arrowprops=dict(arrowstyle='simple',connectionstyle='arc3,rad=0.3',color='gray'))
	ax_conv.annotate('', xy=[0.65,0.39], xytext=[1,0.5],
					size=10, va="center", ha="center",
					xycoords=('figure fraction'), textcoords=(ax_conv,'axes fraction'),
					arrowprops=dict(arrowstyle='simple',connectionstyle='arc3,rad=0',color='gray'))
	
	plt.subplot(gs3[0])
	plt.step(edges[:-1],model_sim_hit_hist,where='pre',color='b',label='Simulation Up')
	plt.plot(m.t,model_teo_rt[0],color='b',linewidth=3,label='Theoretical Up')
	plt.step(edges[:-1],-model_sim_miss_hist,where='pre',color='r',label='Simulation Down')
	plt.plot(m.t,-model_teo_rt[1],color='r',linewidth=3,label='Theoretical Down')
	plt.plot([0,3],[0,0],'-k')
	plt.gca().set_xlim([0,2])
	plt.gca().set_ylim([-0.5*ax1.get_ylim()[-1],ax1.get_ylim()[-1]])
	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().get_xaxis().tick_bottom()
	plt.gca().get_yaxis().tick_left()
	plt.ylabel('RT dist')
	plt.xlabel('T [s]')
	plt.title('Response time')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def bounds_vs_T_n_dt_sketch(fname='bounds_vs_T_n_dt_sketch',suffix='.svg'):
	fname+=suffix
	mo.set_time_units('seconds')
	plt.figure(figsize=(10,4))
	mt.rc('axes', color_cycle=['b','g'])
	gs = gridspec.GridSpec(1, 3,left=0.07, right=0.97,wspace=0.15)
	ax1 = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1],sharey=ax1)
	ax3 = plt.subplot(gs[2],sharey=ax1)
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=4990.24,n=101,T=10,dt=mo.ISI,cost=0.1,reward=1,penalty=0,iti=3,tp=0.)
	xub,xlb,value1,ve1,v11,v22 = m.xbounds(return_values=True)
	rho1 = m.rho
	xb = np.array([xub,xlb])
	plt.sca(ax3)
	plt.plot(m.t,xb[0],linestyle=':',label="dt=40ms",linewidth=2)
	plt.plot(m.t,xb[1],linestyle=':',label="",linewidth=2)
	plt.sca(ax2)
	plt.plot(m.t,xb[0],linestyle=':',label="n=101",linewidth=2)
	plt.plot(m.t,xb[1],linestyle=':',label="",linewidth=2)
	plt.sca(ax1)
	plt.plot(m.t,xb[0],linestyle=':',label="T=10s",linewidth=2)
	plt.plot(m.t,xb[1],linestyle=':',label="",linewidth=2)
	
	plt.sca(ax1)
	m.set_T(20.)
	xb = np.array(m.xbounds())
	plt.plot(m.t,xb[0],linestyle='--',label="T=20s",linewidth=2)
	plt.plot(m.t,xb[1],linestyle='--',label="",linewidth=2)
	m.set_T(30.)
	xb = np.array(m.xbounds())
	plt.plot(m.t,xb[0],linestyle='-',label="T=30s",linewidth=2)
	plt.plot(m.t,xb[1],linestyle='-',label="",linewidth=2)
	plt.xlabel('T [s]')
	plt.ylabel('Bounds')
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	
	plt.sca(ax3)
	m.set_T(10.)
	m.set_dt(0.02)
	xub,xlb,value2,ve2,v11,v22 = m.xbounds(return_values=True)
	rho2 = m.rho
	xb = np.array([xub,xlb])
	plt.plot(m.t,xb[0],linestyle='--',label="dt=20ms",linewidth=2)
	plt.plot(m.t,xb[1],linestyle='--',label="",linewidth=2)
	m.set_dt(0.01)
	xub,xlb,value3,ve3,v11,v22 = m.xbounds(return_values=True)
	rho3 = m.rho
	xb = np.array([xub,xlb])
	plt.sca(ax3)
	plt.plot(m.t,xb[0],linestyle='-',label="dt=10ms",linewidth=2)
	plt.plot(m.t,xb[1],linestyle='-',label="",linewidth=2)
	plt.xlabel('T [s]')
	ax3.tick_params(labelleft=False)
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	
	plt.sca(ax2)
	m.set_dt(mo.ISI)
	m.set_n(51)
	xub,xlb,value2,ve2,v11,v22 = m.xbounds(return_values=True)
	rho2 = m.rho
	xb = np.array([xub,xlb])
	plt.plot(m.t,xb[0],linestyle='--',label="n=51",linewidth=2)
	plt.plot(m.t,xb[1],linestyle='--',label="",linewidth=2)
	m.set_n(25)
	xub,xlb,value3,ve3,v11,v22 = m.xbounds(return_values=True)
	rho3 = m.rho
	xb = np.array([xub,xlb])
	plt.sca(ax2)
	plt.plot(m.t,xb[0],linestyle='-',label="n=25",linewidth=2)
	plt.plot(m.t,xb[1],linestyle='-',label="",linewidth=2)
	plt.xlabel('T [s]')
	ax2.tick_params(labelleft=False)
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	
	place_axes_subfig_label(ax1,'A',horizontal=-0.12,vertical=1.02,fontsize='24')
	place_axes_subfig_label(ax2,'B',horizontal=-0.05,vertical=1.02,fontsize='24')
	place_axes_subfig_label(ax3,'C',horizontal=-0.05,vertical=1.02,fontsize='24')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def vexplore_drop_sketch(fname='vexplore_drop_sketch',suffix='.svg'):
	fname+=suffix
	mo.set_time_units('seconds')
	
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=4990.24,n=101,T=30,dt=mo.ISI,cost=0,reward=1,penalty=0,iti=1.5,tp=0.)
	xub,xlb,v,v_explore,v1,v2 = m.xbounds(return_values=True)
	m.invert_belief()
	p = m.belief_transition_p()
	
	mean_pg = np.sum(p*m.g,axis=2)
	var_pg = np.sum(p*m.g**2,axis=2)-mean_pg**2
	
	plt.figure(figsize=(8,4))
	gs = gridspec.GridSpec(1, 2, left=0.1,right=0.95, wspace=0.28)
	ax2 = plt.subplot(gs[1])
	plt.plot(m.t[:-1],var_pg/var_pg[0],'-',color='k',alpha=0.5)
	plt.xlabel('T [s]')
	plt.ylabel('Normed Transition Var')
	plt.gca().set_ylim([0,1])
	ax1 = plt.subplot(gs[0])
	plt.plot(m.t[:-1],np.mean(v_explore,axis=1),color='k')
	plt.xlabel('T [s]')
	plt.ylabel('V explore')
	place_axes_subfig_label(ax1,'A',horizontal=-0.15,vertical=1.02,fontsize='24')
	place_axes_subfig_label(ax2,'B',horizontal=-0.15,vertical=1.02,fontsize='24')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def bounds_vs_var(fname='bounds_vs_var',suffix='.svg'):
	fname+=suffix
	mo.set_time_units('seconds')
	m = ct.DecisionPolicy(model_var=0.,prior_mu_var=4990.24,n=101,T=10.,dt=mo.ISI,reward=1,penalty=0,iti=1.5,tp=0.)
	
	model_vars = 10.**np.linspace(-4,4,17)
	#~ model_vars = [1e-4,1e-3,1e-2,1e-1,1,1e2,1e3,1e4]
	s1_colors = [plt.get_cmap('YlGn')(x) for x in np.linspace(0,1,len(model_vars))]
	s2_colors = [plt.get_cmap('YlOrRd')(x) for x in np.linspace(0,1,len(model_vars))]
	#~ s1_colors = [plt.get_cmap('brg')(x) for x in np.linspace(0,0.5,len(model_vars))]
	#~ s2_colors = [plt.get_cmap('brg')(x) for x in np.linspace(1,0.5,len(model_vars))]
	mus = np.array([0.1,0.5,1.,2.,5.,10.,15.,25.,50.,100.])
	
	plt.figure(figsize=(14,7))
	gs1 = gridspec.GridSpec(2, 1,left=0.1, right=0.5,hspace=0.08)
	gs2 = gridspec.GridSpec(1, 1,left=0.505, right=0.52)
	gs3 = gridspec.GridSpec(1, 1,left=0.65, right=0.95)
	ax1 = plt.subplot(gs1[0])
	ax2 = plt.subplot(gs1[1])
	performance = np.zeros((len(model_vars),len(mus)))
	evidence = np.zeros_like(performance)
	for i,(v,c1,c2) in enumerate(zip(model_vars,s1_colors,s2_colors)):
		m.set_internal_var(v)
		xub,xlb = m.xbounds()
		ax1.plot(m.t,xub,color=c1)
		ax1.plot(m.t,xlb,color=c2)
		ax2.plot(m.t,m.bounds[0],color=c1)
		ax2.plot(m.t,m.bounds[1],color=c2)
		for j,mu in enumerate(mus):
			rt = m.rt(mu,bounds=(xub,xlb))
			evidence[i,j] = mu/np.sqrt(v)
			performance[i,j] = np.sum(rt[0])*m.dt
	ax1.set_ylabel('$x(t)$ Bounds',fontsize=16)
	ax1.tick_params(labelbottom=False)
	ax2.set_ylabel('$g$ Bounds',fontsize=16)
	ax2.set_xlabel('T [s]',fontsize=16)
	
	cbar_ax = plt.subplot(gs2[0])
	cbar = np.array([[plt.get_cmap('YlGn')(x) for x in np.linspace(0,1,100)],
					 [plt.get_cmap('YlOrRd')(x) for x in np.linspace(0,1,100)]])
	cbar = np.swapaxes(cbar,0,1)
	plt.sca(cbar_ax)
	cbar_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(r'$10^{%1.0f}$'))
	cbar_ax.yaxis.set_label_position("right")
	cbar_ax.tick_params(reset=True,which='major',axis='y',direction='in',left=True, right=True,bottom=False,top=False,labelleft=False, labelright=True)
	plt.imshow(cbar,aspect='auto',cmap=None,interpolation='none',origin='lower',extent=[0,1,-4,4])
	cbar_ax.xaxis.set_ticks([])
	plt.ylabel('$\sigma^{2}$ [Hz]',fontsize=16)
	
	evidence = evidence.flatten()
	performance = performance.flatten()
	if any(np.isnan(performance)):
		evidence = evidence[np.logical_not(np.isnan(performance))]
		performance = performance[np.logical_not(np.isnan(performance))]
	fun = lambda x,a: 1./(1.+np.exp(a*x))
	from scipy.optimize import curve_fit
	popt,pcov = curve_fit(fun, evidence, performance)
	try:
		print('Par = {0}+-{1}'.format(popt[0],np.sqrt(pcov[0][0])))
	except:
		print('Par = {0}+-{1}'.format(popt,np.sqrt(pcov)))
	
	ax = plt.subplot(gs3[0])
	plt.plot(evidence,performance,'o',label='Simulations',color='k')
	x = np.linspace(0,100,10000)
	plt.plot(x,fun(x,popt),'-r',label=r'$1/\left[1+\exp\left(%1.4f x\right)\right]$'%(popt))
	plt.xlim([0,10])
	plt.ylabel('Performance',fontsize=16)
	plt.xlabel(r'$\mu/\sigma$',fontsize=18)
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	
	place_axes_subfig_label(ax1,'A',horizontal=-0.07,fontsize='24')
	place_axes_subfig_label(ax2,'B',horizontal=-0.08,fontsize='24')
	place_axes_subfig_label(ax,'C',horizontal=-0.1,vertical=1.02,fontsize='24')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def performance_vs_var_and_cost(fname='performance_vs_var_and_cost',suffix='.svg'):
	fname+=suffix
	mo.set_time_units('seconds')
	m = ct.DecisionPolicy(model_var=0.,prior_mu_var=4990.24,n=101,T=10.,dt=mo.ISI,reward=1,penalty=0,iti=1.5,tp=0.)
	model_vars = [1e-4,1e-3,1e-2,1e-1,1,1e2,1e3,1e4]
	costs = np.linspace(0,0.5,10)
	#~ s1_colors = [plt.get_cmap('YlGn')(x) for x in np.linspace(0,1,len(model_vars))]
	#~ s2_colors = [plt.get_cmap('YlOrRd')(x) for x in np.linspace(0,1,len(model_vars))]
	mus = np.array([0.1,0.5,1.])
	
	plt.figure(figsize=(14,7))
	#~ gs1 = gridspec.GridSpec(2, 1,left=0.1, right=0.5,hspace=0.08)
	#~ gs2 = gridspec.GridSpec(1, 1,left=0.505, right=0.52)
	#~ gs3 = gridspec.GridSpec(1, 1,left=0.65, right=0.95)
	#~ ax1 = plt.subplot(gs1[0])
	#~ ax2 = plt.subplot(gs1[1])
	performance = np.zeros((len(mus),len(costs),len(model_vars)))
	evidence = np.zeros_like(performance)
	for k,mu in enumerate(mus):
		print(mu)
		for i,cost in enumerate(costs):
			m.set_cost(cost)
			for j,var in enumerate(model_vars):
				m.set_internal_var(var)
				xub,xlb = m.xbounds()
				performance[k,i,j] = np.sum(m.rt(mu,bounds=(xub,xlb))[0])*m.dt
		plt.subplot(1,3,k+1)
		plt.plot(model_vars,performance[k].T)#,aspect='auto',cmap=None,interpolation='none',origin='lower')
		plt.xlabel(r'$\sigma$')
	plt.show()

def cluster_hierarchy(fname='cluster_hierarchy',suffix='.svg'):
	fname+=suffix
	analysis.cluster_analysis(method='full_confidence', optimizer='cma', suffix='', override=False,\
				affinity='euclidean', linkage='ward', pooling_func=np.nanmean,\
				merge='names',filter_nans='post', tree_mode='r',extension='png')
	a = analysis.Analyzer(method='full_confidence', optimizer='cma', suffix='',\
							override=False, affinity='euclidean', linkage='ward', pooling_func=np.nanmean)
	a.get_parameter_array_from_summary(normalize={'internal_var':'experiment',\
												  'confidence_map_slope':'all',\
												  'cost':'all',\
												  'high_confidence_threshold':'all',\
												  'dead_time':'all',\
												  'dead_time_sigma':'all',\
												  'phase_out_prob':'all'})
	
	decision_parameters=['cost','internal_var','phase_out_prob']
	confidence_parameters=['high_confidence_threshold','confidence_map_slope']
	
	fig = plt.figure(figsize=(13,10))
	gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[8,7])
	ax1 = fig.add_subplot(gs[0,0],projection='3d')
	ax2 = fig.add_subplot(gs[0,1])
	ax3 = fig.add_subplot(gs[1,0])
	ax4 = fig.add_subplot(gs[1,1])
	
	a.controlled_scatter(axes=ax1,scattered_parameters=decision_parameters,merge='names')
	a.controlled_scatter(axes=ax2,scattered_parameters=confidence_parameters,merge='names')
	ax1.set_xlabel(ax1.get_xlabel(),fontsize=22)
	ax1.set_ylabel(ax1.get_ylabel(),fontsize=22)
	ax1.set_zlabel(ax1.get_zlabel(),fontsize=22)
	ax2.set_xlabel(ax2.get_xlabel(),fontsize=22)
	ax2.set_ylabel(ax2.get_ylabel(),fontsize=22)
	
	ax2.legend(loc='upper left', fancybox=True, framealpha=0.5, fontsize=15)
	decision_hierarchy = mpimg.imread('../../figs/decision_cluster.png')
	confidence_hierarchy = mpimg.imread('../../figs/confidence_cluster.png')
	ax3.imshow(decision_hierarchy)
	ax3.set_axis_off()
	ax4.imshow(confidence_hierarchy)
	ax4.set_axis_off()
	place_axes_subfig_label(ax1,'A',horizontal=0.16,vertical=1.02,fontsize='30')
	place_axes_subfig_label(ax2,'B',horizontal=-0.05,vertical=1.02,fontsize='30')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def confidence_mapping(fname='confidence_mapping',suffix='.svg'):
	fname+=suffix
	
	subjects = io_cog.filter_subjects_list(io_cog.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment')
	subjects = io_cog.merge_subjectSessions(io_cog.filter_subjects_list(subjects,'experiment_Luminancia'),merge='all')
	subject = subjects[0]
	
	fitter = fits.Fitter(subject,method='full_cognition',decisionPolicyKwArgs={'n':201,'dt':1e-3,'T':1.})
	parameters = fitter.get_parameters_dict()
	parameters['high_confidence_threshold'] = 1.2
	parameters['dead_time_sigma'] = 0.3
	bin_map = fitter.high_confidence_mapping(parameters['high_confidence_threshold'],np.inf)
	cont_map = fitter.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
	plt.figure(figsize=(14,6))
	ax1 = plt.subplot(121)
	plt.plot(fitter.dp.t,bin_map[0],'r',label='Binary',linewidth=2)
	plt.plot(fitter.dp.t,cont_map[0],'b',label='Continuous',linewidth=2)
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	plt.gca().set_xlim([0,0.4])
	plt.xlabel('T [s]')
	plt.ylabel(r'$M(C(\theta_{+}(t)))$',fontsize='13')
	xytext_bin = [fitter.dp.t[(bin_map[0]==0).nonzero()[0][0]],0.8]
	xytext_cont = [fitter.dp.t[np.argmin(np.abs(cont_map[0]-0.2))+2],0.2]
	
	fpt = fitter.dp.rt(0,fitter.dp.xbounds())
	bin_pdf = fitter.rt_confidence_pdf(fpt,parameters,bin_map)
	t = np.arange(0,bin_pdf.shape[-1],dtype=np.float)*fitter.dp.dt
	
	gs = gridspec.GridSpec(2, 1,left=0.54, right=0.94,hspace=0.25)
	ax2 = plt.subplot(gs[0])
	#~ plt.imshow(bin_pdf[0],aspect='auto',interpolation='none',origin='lower',extent=[0,t[-1],0,1])
	plt.plot(t,bin_pdf[0][-1],color='forestgreen',label='high confidence',linewidth=2)
	plt.plot(t,bin_pdf[0][0],color='mediumpurple',label='low confidence',linewidth=2)
	plt.ylabel('Prob density')
	plt.xlabel('T [s]')
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	
	vmin = 1e-5
	
	cont_pdf = fitter.rt_confidence_pdf(fpt,parameters,cont_map)
	ax3 = plt.subplot(gs[1])
	if not vmin is None:
		cont_pdf[cont_pdf<vmin] = vmin
	plt.imshow(cont_pdf[0],aspect='auto',interpolation='none',origin='lower',extent=[0,t[-1],0,1],\
				vmin=vmin,cmap=plt.get_cmap('gray_r'),norm=LogNorm())
	plt.xlabel('T [s]')
	plt.ylabel(r'$M(C(g))$',fontsize='13')
	cbar = plt.colorbar()
	cbar.set_label('Prob density')
	
	ax1.annotate('', xy=[-0.12,0.8], xytext=xytext_bin,
				size=10, va="center", ha="center",
				xycoords=(ax2,'axes fraction'), textcoords=('data'),
				arrowprops=dict(arrowstyle='simple',connectionstyle='arc3,rad=0',color='black'))
	
	ax1.annotate('', xy=[-0.17,0.2], xytext=xytext_cont,
				size=10, va="center", ha="center",
				xycoords=(ax3,'axes fraction'), textcoords=('data'),
				arrowprops=dict(arrowstyle='simple',connectionstyle='arc3,rad=0',color='black'))
	
	place_axes_subfig_label(ax1,'A',horizontal=-0.08,vertical=1.05,fontsize='24')
	place_axes_subfig_label(ax2,'B',horizontal=-0.08,vertical=1.12,fontsize='24')
	place_axes_subfig_label(ax3,'C',horizontal=-0.11,vertical=1.12,fontsize='24')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def fits_cognition(fname='fits_cognition',suffix='.svg'):
	fname+=suffix
	subjects = io_cog.filter_subjects_list(io_cog.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment')
	fitter_plot_handler = None
	for i,s in enumerate(subjects):
		handler_fname = fits.Fitter_filename(experiment=s.experiment,method='full_confidence',name=s.get_name(),
				session=s.get_session(),optimizer='cma',suffix='').replace('.pkl','_plot_handler.pkl')
		try:
			f = open(handler_fname,'r')
			temp = pickle.load(f)
			f.close()
		except:
			continue
		if fitter_plot_handler is None:
			fitter_plot_handler = temp
		else:
			fitter_plot_handler+= temp
	fitter_plot_handler = fitter_plot_handler.merge('all')
	
	experiment_alias = {'2AFC':'Contrast','Auditivo':'Auditory','Luminancia':'Luminance'}
	row_index = {'2AFC':1,'Auditivo':2,'Luminancia':3}
	plot_gs = gridspec.GridSpec(3, 2,left=0.25, right=0.98,hspace=0.20,wspace=0.18)
	exp_scheme_gs = gridspec.GridSpec(3, 1,left=0., right=0.16,hspace=0.25)
	plt.figure(figsize=(12,11))
	fitter_plot_handler.normalize()
	for experiment in fitter_plot_handler.keys():
		subj = fitter_plot_handler[experiment]['experimental']
		model = fitter_plot_handler[experiment]['theoretical']
		row = row_index[experiment]
		exp_name = experiment_alias[experiment]
		
		axrt = plt.subplot(plot_gs[row-1,0])
		dt = subj['t_array'][1]-subj['t_array'][0]
		dc = subj['c_array'][1]-subj['c_array'][0]
		if len(subj['t_array'])==len(subj['rt'][0]):
			subj_t_array = np.hstack((subj['t_array']-0.5*dt,subj['t_array'][-1:]+0.5*dt))
			subj_c_array = np.hstack((subj['c_array']-0.5*dc,subj['c_array'][-1:]+0.5*dc))
		else:
			subj_t_array = subj['t_array']
			subj_c_array = subj['c_array']
		subj_rt = np.hstack((subj['rt'],np.array([subj['rt'][:,-1]]).T))
		subj_confidence = np.hstack((subj['confidence'],np.array([subj['confidence'][:,-1]]).T))
		plt.step(subj_t_array,subj_rt[0],'b',label='Subject hit',where='post')
		plt.step(subj_t_array,subj_rt[1],'r',label='Subject miss',where='post')
		plt.plot(model['t_array'],model['rt'][0],'b',label='Model hit',linewidth=3)
		plt.plot(model['t_array'],model['rt'][1],'r',label='Model miss',linewidth=3)
		axrt.set_xlim([0,subj['t_array'][-1]+0.5*(subj['t_array'][-1]-subj['t_array'][-2])])
		if row==1:
			plt.title('RT distribution')
			plt.legend(loc='best', fancybox=True, framealpha=0.5)
		elif row==3:
			plt.xlabel('T [s]')
		plt.ylabel('Prob density')
		
		axconf = plt.subplot(plot_gs[row-1,1])
		plt.step(subj_c_array,subj_confidence[0],'b',label='Subject hit',where='post')
		plt.step(subj_c_array,subj_confidence[1],'r',label='Subject miss',where='post')
		plt.plot(model['c_array'],model['confidence'][0],'b',label='Model hit',linewidth=3)
		plt.plot(model['c_array'],model['confidence'][1],'r',label='Model miss',linewidth=3)
		axconf.set_yscale('log')
		if row==1:
			plt.title('Confidence distribution')
		elif row==3:
			plt.xlabel('Confidence')
		
		aximg = plt.subplot(exp_scheme_gs[row-1])
		
		exp_scheme = mpimg.imread('../../figs/'+experiment+'.png')
		aximg.imshow(exp_scheme)
		aximg.set_axis_off()
		plt.title(exp_name)
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def binary_confidence(fname='binary_confidence',suffix='.svg'):
	fname+=suffix
	subjects = io_cog.filter_subjects_list(io_cog.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment')
	fitter_plot_handler = None
	binary_split_method = 'mean'
	suffix = '' if binary_split_method=='median' else '_'+binary_split_method+'split'
	for i,s in enumerate(subjects):
		handler_fname = fits.Fitter_filename(experiment=s.experiment,method='full_binary_confidence',
				name=s.get_name(),session=s.get_session(),optimizer='basinhopping',
				suffix=suffix,confidence_map_method='belief').replace('.pkl','_plot_handler.pkl')
		try:
			f = open(handler_fname,'r')
			temp = pickle.load(f)
			f.close()
		except:
			continue
		if fitter_plot_handler is None:
			fitter_plot_handler = temp
		else:
			fitter_plot_handler+= temp
	fitter_plot_handler = fitter_plot_handler.merge('all')
	
	experiment_alias = {'2AFC':'Contrast','Auditivo':'Auditory','Luminancia':'Luminance'}
	row_index = {'2AFC':1,'Auditivo':2,'Luminancia':3}
	plot_gs = gridspec.GridSpec(3, 3,left=0.20, right=0.98,hspace=0.20,wspace=0.18)
	exp_scheme_gs = gridspec.GridSpec(3, 1,left=0., right=0.14,hspace=0.25)
	plt.figure(figsize=(14,11))
	fitter_plot_handler.normalize()
	for experiment in fitter_plot_handler.keys():
		subj = fitter_plot_handler[experiment]['experimental']
		model = fitter_plot_handler[experiment]['theoretical']
		row = row_index[experiment]
		exp_name = experiment_alias[experiment]
		
		dt = subj['t_array'][1]-subj['t_array'][0]
		subj_performance = np.sum(subj['rt'][0])*dt
		if binary_split_method=='median':
			subj_split_ind = (np.cumsum(np.sum(subj['confidence'],axis=0))>=0.5).nonzero()[0][0]
			if model['confidence'].shape[1]>2:
				self.logger.debug('Model confidence data is not natively binary. Binarizing now...')
				model_split_ind = (np.cumsum(np.sum(model['confidence'],axis=0))>=0.5).nonzero()[0][0]
			else:
				model_split_ind = 1
		elif binary_split_method=='half':
			subj_split_ind = (subj['c_array']>=0.5).nonzero()[0][0]
			if model['confidence'].shape[1]>2:
				self.logger.debug('Model confidence data is not natively binary. Binarizing now...')
				model_split_ind = (model['c_array']>=0.5).nonzero()[0][0]
			else:
				model_split_ind = 1
		elif binary_split_method=='mean':
			if len(subj['c_array'])>(subj['confidence'].shape[1]):
				c_array = np.array([0.5*(e1+e0) for e1,e0 in zip(subj['c_array'][1:],subj['c_array'][:-1])])
			else:
				c_array = subj['c_array']
			subj_split_ind = (c_array>=np.sum(subj['confidence']*c_array)).nonzero()[0][0]
			if model['confidence'].shape[1]>2:
				self.logger.debug('Model confidence data is not natively binary. Binarizing now...')
				model_split_ind = (model['c_array']>=np.sum(model['confidence']*model['c_array'])).nonzero()[0][0]
			else:
				model_split_ind = 1
		subj_lowconf_rt = np.array([np.sum(subj['hit_histogram'][:subj_split_ind],axis=0)*subj_performance,
									np.sum(subj['miss_histogram'][:subj_split_ind],axis=0)*(1-subj_performance)])
		subj_highconf_rt = np.array([np.sum(subj['hit_histogram'][subj_split_ind:],axis=0)*subj_performance,
									np.sum(subj['miss_histogram'][subj_split_ind:],axis=0)*(1-subj_performance)])
		model_performance = np.sum(model['rt'][0])*(model['t_array'][1]-model['t_array'][0])
		model_low_rt = np.array([np.sum(model['hit_histogram'][:model_split_ind],axis=0)*model_performance,
								np.sum(model['miss_histogram'][:model_split_ind],axis=0)*(1-model_performance)])
		model_high_rt = np.array([np.sum(model['hit_histogram'][model_split_ind:],axis=0)*model_performance,
								np.sum(model['miss_histogram'][model_split_ind:],axis=0)*(1-model_performance)])
		if len(subj['t_array'])==len(subj['rt'][0]):
			subj_t_array = np.hstack((subj['t_array']-0.5*dt,subj['t_array'][-1:]+0.5*dt))
		else:
			subj_t_array = subj['t_array']
		subj_rt = np.hstack((subj['rt'],np.array([subj['rt'][:,-1]]).T))
		subj_lowconf_rt = np.hstack((subj_lowconf_rt,np.array([subj_lowconf_rt[:,-1]]).T))
		subj_highconf_rt = np.hstack((subj_highconf_rt,np.array([subj_highconf_rt[:,-1]]).T))
		
		axrt = plt.subplot(plot_gs[row-1,0])
		plt.step(subj_t_array,subj_rt[0],'b',label='Subject P(RT,hit)',where='post')
		plt.step(subj_t_array,subj_rt[1],'r',label='Subject P(RT,miss)',where='post')
		plt.plot(model['t_array'],model['rt'][0],'b',label='Model P(RT,hit)',linewidth=3)
		plt.plot(model['t_array'],model['rt'][1],'r',label='Model P(RT,miss)',linewidth=3)
		axrt.set_xlim([0,subj['t_array'][-1]+0.5*(subj['t_array'][-1]-subj['t_array'][-2])])
		plt.ylabel('Prob density')
		if row==1:
			plt.title('RT distribution')
			plt.legend(loc='best', fancybox=True, framealpha=0.5)
		elif row==3:
			plt.xlabel('T [s]')
		plt.ylabel('Prob density')
		
		axconf = plt.subplot(plot_gs[row-1,1])
		plt.step(subj_t_array,np.sum(subj_lowconf_rt,axis=0),'mediumpurple',label='Subject P(RT,low)',where='post')
		plt.step(subj_t_array,np.sum(subj_highconf_rt,axis=0),'forestgreen',label='Subject P(RT,high)',where='post')
		plt.plot(model['t_array'],np.sum(model_low_rt,axis=0),'mediumpurple',label='Model P(RT,low)',linewidth=3)
		plt.plot(model['t_array'],np.sum(model_high_rt,axis=0),'forestgreen',label='Model P(RT,high)',linewidth=3)
		axconf.set_xlim([0,subj['t_array'][-1]+0.5*(subj['t_array'][-1]-subj['t_array'][-2])])
		if row==1:
			plt.title('Confidence distribution')
			plt.legend(loc='best', fancybox=True, framealpha=0.5)
		elif row==3:
			plt.xlabel('Confidence')
		
		mdt = model['t_array'][1]-model['t_array'][0]
		axconf2 = plt.subplot(plot_gs[row-1,2])
		plt.step(subj_t_array,np.sum(subj_lowconf_rt,axis=0)/np.sum(subj_lowconf_rt*dt),'mediumpurple',label='Subject P(RT|low)',where='post')
		plt.step(subj_t_array,np.sum(subj_highconf_rt,axis=0)/np.sum(subj_highconf_rt*dt),'forestgreen',label='Subject P(RT|high)',where='post')
		plt.plot(model['t_array'],np.sum(model_low_rt,axis=0)/np.sum(model_low_rt*mdt),'mediumpurple',label='Model P(RT|low)',linewidth=3)
		plt.plot(model['t_array'],np.sum(model_high_rt,axis=0)/np.sum(model_high_rt*mdt),'forestgreen',label='Model P(RT|high)',linewidth=3)
		axconf2.set_xlim([0,subj['t_array'][-1]+0.5*(subj['t_array'][-1]-subj['t_array'][-2])])
		if row==1:
			plt.title('Conditional confidence distribution')
			plt.legend(loc='best', fancybox=True, framealpha=0.5)
		elif row==3:
			plt.xlabel('Confidence')
		
		aximg = plt.subplot(exp_scheme_gs[row-1])
		
		exp_scheme = mpimg.imread('../../figs/'+experiment+'.png')
		aximg.imshow(exp_scheme)
		aximg.set_axis_off()
		plt.title(exp_name)
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def auxiliary_2AFC_stimuli():
	x = np.array([np.linspace(-1,1,1000)])
	y = x.copy().T
	n1 = np.random.randint(0,2,(x.size,y.size))
	n2 = np.random.randint(0,2,(x.size,y.size))
	n3 = np.random.randint(0,2,(x.size,y.size))
	freq = 3.*np.pi
	ori = -30.*np.pi/180.
	grating = np.sin(freq*(np.sin(ori)*x+y))
	mask = np.sqrt(x**2+y**2)<=1.
	alpha = 0.15

	target = alpha*grating+(1-alpha)*n1
	distractor = alpha*n2+(1-alpha)*n3
	target[np.logical_not(mask)] = np.nan
	distractor[np.logical_not(mask)] = np.nan

	plt.figure()
	plt.imshow(target,cmap=plt.get_cmap('gray'))
	plt.gca().set_axis_off()
	plt.savefig('../../figs/2AFC_target.png',bbox_inches='tight',dpi=200, transparent=True, pad_inches=0.)
	plt.figure()
	plt.imshow(distractor,cmap=plt.get_cmap('gray'))
	plt.gca().set_axis_off()
	plt.savefig('../../figs/2AFC_distractor.png',bbox_inches='tight',dpi=200, transparent=True, pad_inches=0.)

def parameter_correlation(fname='parameter_correlation',suffix='.svg'):
	fname+=suffix
	a = analysis.Analyzer()
	parameter_names = ['cost','internal_var','phase_out_prob','high_confidence_threshold','confidence_map_slope']
	stats_names = ['rt_mean','performance_mean','confidence_mean','auc','multi_mod_index']
	parameter_aliases = {'cost':r'$c$',
						'internal_var':r'$\sigma^{2}$',
						'phase_out_prob':r'$p_{po}$',
						'high_confidence_threshold':r'$C_{H}$',
						'confidence_map_slope':r'$\alpha$',
						'dead_time':r'$\tau_{c}$',
						'dead_time_sigma':r'$\sigma_{c}$'}
	stats_aliases = {'rt_mean':'Mean RT',
					'performance_mean':'Performance',
					'confidence_mean':'Mean Confidence',
					'auc':'AUC',
					'multi_mod_index':"Hartigan's dip"}
	method = 'spearman'
	corrs1,pvals1 = a.parameter_correlation(parameter_names,method=method,correct_multiple_comparison_pvalues=True)
	corrs2,pvals2 = a.correlate_parameters_with_stats(stats_names,parameter_names,method=method,correct_multiple_comparison_pvalues=True)
	corrs1[pvals1>0.05] = np.nan
	annotations = []
	for pval in pvals2:
		t = []
		for p in pval:
			mark = ''
			if p<0.05:
				mark+= '*'
				if p<0.005:
					mark+= '*'
					if p<0.0005:
						mark+= '*'
			t.append(mark)
		annotations.append(t)
	plt.figure(figsize=(14,8))
	ax2 = plt.subplot(121)
	grouped_bar_plot(corrs2,ax=ax2,annotations=annotations,#colors=['r','g','b','y','m'],
					group_member_names=[stats_aliases[s] for s in stats_names],
					group_names=[parameter_aliases[p] for p in parameter_names])
	[tick.label.set_fontsize(16) for tick in ax2.xaxis.get_major_ticks()]
	plt.ylabel('Correlation')
	plt.subplot(122)
	plt.imshow(corrs1,aspect='auto',cmap='seismic',interpolation='none',extent=[0,len(corrs1),0,len(corrs1)],vmin=-1,vmax=1)
	plt.xticks(np.arange(len(corrs1))+0.5,[parameter_aliases[p] for p in parameter_names],rotation=60,fontsize=14)
	plt.yticks(np.arange(len(corrs1))+0.5,[parameter_aliases[p] for p in parameter_names][::-1],fontsize=14)
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('Correlation',fontsize=14)
	plt.show(True)
	#~ plt.savefig('../../figs/'+fname,bbox_inches='tight')

def parse_input():
	script_help = """ figures.py help
 Sintax:
 figures.py [figure flags] [--show] [--suffix suffix_value]
 
 figures.py -h [or --help] displays help
 
 Optional arguments are:
 '--all': If present plots all figures
 '--bounds_vs_cost': Plot bounds_vs_cost
 '--rt_fit': Plot rt distribution fits
 '--value_and_bounds_sketch': Plot value and bounds as a function of time and g sketch
 '--confidence_sketch': Plot confidence report sketch
 '--decision_rt_sketch': Plot decision rt consolidation sketch
 '--bounds_vs_T_n_dt_sketch': Plot bounds as a function of dt and T sketch
 '--vexplore_drop_sketch': Plot variance of p(g+dg|g) as a function of time and also plot the drop in mean v_explore
 '--bounds_vs_var': Plot decision bounds as a function of model_var
 '--performance_vs_var_and_cost'
 '--cluster_hierarchy'
 '--confidence_mapping'
 '--fits_cognition'
 '--parameter_correlation'
 
 '--show': Show the matplotlib figures after all have been created
 '--suffix': The suffix to append at the end of the figure filenames [Default = '.svg']
 """
	options =  {'bounds_vs_cost':False,'rt_fit':False,'value_and_bounds_sketch':False,
				'confidence_sketch':False,'decision_rt_sketch':False,'bounds_vs_T_n_dt_sketch':False,
				'prior_sketch':False,'vexplore_drop_sketch':False,'show':False,'suffix':'.svg',
				'bounds_vs_var':False,'performance_vs_var_and_cost':False,'cluster_hierarchy':False,
				'confidence_mapping':False,'fits_cognition':False,'binary_confidence':False,
				'parameter_correlation':False}
	keys = options.keys()
	skip_arg = False
	for i,arg in enumerate(sys.argv[1:]):
		if skip_arg:
			skip_arg = False
			continue
		if arg=='-h' or arg=='--help':
			print(script_help)
			sys.exit()
		elif arg=='--all':
			for key in keys:
				if not (key=='show' or key=='suffix'):
					options[key] = True
		elif arg[2:]=='suffix':
			options['suffix'] = sys.argv[i+2]
			skip_arg = True
		elif arg[2:] in keys:
			options[arg[2:]] = True
		else:
			raise RuntimeError("Unknown option: {opt} encountered in position {pos}. Refer to the help to see the list of options".format(opt=arg,pos=i+1))
	return options

if __name__=="__main__":
	parameter_correlation()
	opts = parse_input()
	for k in opts.keys():
		if k not in ['suffix','show'] and opts[k]:
			eval(k+"(suffix=opts['suffix'])")
	
	if opts['show']:
		plt.show(True)
