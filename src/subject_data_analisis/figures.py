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

def value_and_bounds_sketch(fname='value_and_bounds_sketch.svg'):
	mo.set_time_units('seconds')
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=4990.24,n=101,T=30,dt=0.01,reward=1,penalty=0,iti=1.5,tp=0.)
	m.cost = 0.1
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
	plt.plot(m.g,np.max(np.array([v1,v2]),axis=0),linestyle='--',color='k',linewidth=3)
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
	plt.plot(m.t[m.t<3],b[:,m.t<3].T,linewidth=2)
	plt.ylabel('$g$ bound')
	plt.gca().set_ylim([0,1])
	# Plot decision bounds in the diffusion space
	plt.subplot(224)
	plt.plot(m.t[m.t<3],xb[:,m.t<3].T,linewidth=2)
	
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
	
	plt.savefig('../../figs/'+fname)

def bounds_vs_cost(fname='bounds_cost.svg',n_costs=20,maxcost=1.,prior_mu_var=4990.24,n=101,T=10.,dt=None,reward=1,penalty=0,iti=1.5,tp=0.):
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
		m.cost = cost
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

def rt_fit(fname='rt_fit.svg'):
	subjects = io.unique_subjects(mo.data_dir)
	files = ['fits/inference_fit_full_subject_'+str(sid)+'_seconds.pkl' for sid in range(1,7)]
	files2 = ['fits/inference_fit_confidence_only_subject_'+str(sid)+'_seconds.pkl' for sid in range(1,7)]
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
		
		m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=prior_mu_var,n=n,T=T,dt=dt,reward=reward,penalty=penalty,iti=iti,tp=tp,store_p=False)
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
	plt.figure(figsize=(11,8))
	ax1 = plt.subplot(121)
	plt.step(xh,hit_rt,label='Subjects hit rt',where='post',color='b')
	plt.step(xh,-miss_rt,label='Subjects miss rt',where='post',color='r')
	plt.plot(m.t,all_sim_rt['hit'],label='Theoretical hit rt',linewidth=2,color='b')
	plt.plot(m.t,-all_sim_rt['miss'],label='Theoretical miss rt',linewidth=2,color='r')
	plt.xlim([0,mxlim])
	if options['time_units']=='seconds':
		plt.xlabel('T [s]')
	else:
		plt.xlabel('T [ms]')
	plt.ylabel('Prob density')
	plt.legend()
	# Plot confidence data
	plt.subplot(122,sharey=ax1)
	plt.step(xh,high_hit_rt+high_miss_rt,label='Subjects high',where='post',color='forestgreen')
	plt.step(xh,-(low_hit_rt+low_miss_rt),label='Subjects low',where='post',color='mediumpurple')
	plt.plot(m.t,all_sim_rt['high'],label='Theoretical high',linewidth=2,color='forestgreen')
	plt.plot(m.t,-all_sim_rt['low'],label='Theoretical low',linewidth=2,color='mediumpurple')
	plt.xlim([0,mxlim])
	if options['time_units']=='seconds':
		plt.xlabel('T [s]')
	else:
		plt.xlabel('T [ms]')
	plt.legend()
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def prior_sketch(fname='prior_sketch.svg'):
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
	plt.legend()
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def confidence_sketch(fname='confidence_sketch.svg'):
	mo.set_time_units('seconds')
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=4990.24,n=101,T=10.,dt=mo.ISI,reward=1.,penalty=0.,iti=1.5,tp=0.,store_p=False)
	cost = 0.01
	dead_time = 0.3
	dead_time_sigma = 0.4
	phase_out_prob = 0.05
	xub,xlb = m.xbounds()
	log_odds = m.log_odds()
	high_conf_threshold = 1.
	ind_bound = (log_odds[0]<=high_conf_threshold).nonzero()[0][0]
	
	plt.figure(figsize=(8,6))
	ax1=plt.subplot(211)
	p0, = plt.plot(m.t,log_odds[0],color='k',linewidth=3,label=r'$C(t)$')
	plt.plot([0,3],high_conf_threshold*np.ones(2),'--k')
	p1 = plt.Rectangle((0, 0), 1, 1, fc="forestgreen")
	p2 = plt.Rectangle((0, 0), 1, 1, fc="mediumpurple")
	plt.fill_between(m.t[:ind_bound+1],log_odds[0][:ind_bound+1],interpolate=True,color='forestgreen',alpha=0.6)
	plt.fill_between(m.t[ind_bound:],log_odds[0][ind_bound:],interpolate=True,color='mediumpurple',alpha=0.6)
	plt.legend([p0,p1, p2], [r'$C(t)$','High confidence zone', 'Low confidence zone'])
	plt.ylabel('Log odds')
	ax1.tick_params(labelleft=True,labelbottom=False)
	
	plt.subplot(212,sharex=ax1)
	plt.plot(m.t,xub,color='b')
	plt.plot(m.t,xlb,color='r')
	plt.fill_between(m.t[:ind_bound+1],xub[:ind_bound+1],xlb[:ind_bound+1],interpolate=True,color='forestgreen',alpha=0.6)
	plt.fill_between(m.t[ind_bound:],xub[ind_bound:],xlb[ind_bound:],interpolate=True,color='mediumpurple',alpha=0.6)
	ax1.set_xlim([0,3])
	plt.ylabel(r'Bound in $x(t)$ space')
	plt.xlabel('T [s]')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def decision_rt_sketch(fname='decision_rt_sketch.svg'):
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
	
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=prior_mu_var,n=101,T=10.,dt=mo.ISI,reward=1.,penalty=0.,iti=1.5,tp=0.,store_p=False)
	cost = 0.01
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
	plt.legend()
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
	ax2.set_ylim([-np.ceil(max_b/10)*10,np.ceil(max_b/10)*10])
	
	ndec = [1,1]
	ax2 = plt.subplot(gs1[1,0],sharex=ax1)
	for sample in reversed(samples):
		if sample['dec'] and sample['rt']>=0.2:
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
	plt.gca().set_xlim([0,2])
	
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
	plt.gca().set_xlim([0,3])
	plt.gca().set_ylim([-0.5*ax1.get_ylim()[-1],ax1.get_ylim()[-1]])
	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().get_xaxis().tick_bottom()
	plt.gca().get_yaxis().tick_left()
	plt.ylabel('RT dist')
	plt.xlabel('T [s]')
	plt.title('Response time')
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def bounds_vs_T_n_dt_sketch(fname='bounds_vs_T_n_dt_sketch.svg'):
	mo.set_time_units('seconds')
	plt.figure(figsize=(10,4))
	mt.rc('axes', color_cycle=['b','g'])
	ax1 = plt.subplot(131)
	ax2 = plt.subplot(132,sharey=ax1)
	ax3 = plt.subplot(133,sharey=ax1)
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=4990.24,n=101,T=10,dt=mo.ISI,reward=1,penalty=0,iti=1.5,tp=0.)
	m.cost = 0.1
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
	plt.legend()
	
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
	plt.legend()
	
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
	plt.legend()
	
	plt.savefig('../../figs/'+fname,bbox_inches='tight')

def vexplore_drop_sketch(fname='vexplore_drop_sketch.svg'):
	mo.set_time_units('seconds')
	
	m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=4990.24,n=101,T=30,dt=mo.ISI,reward=1,penalty=0,iti=1.5,tp=0.)
	m.cost = 0.
	xub,xlb,v,v_explore,v1,v2 = m.xbounds(return_values=True)
	m.invert_belief()
	p = m.belief_transition_p()
	
	mean_pg = np.sum(p*m.g,axis=2)
	var_pg = np.sum(p*m.g**2,axis=2)-mean_pg**2
	
	plt.figure(figsize=(8,4))
	gs = gridspec.GridSpec(1, 2, left=0.1,right=0.95, wspace=0.28)
	plt.subplot(gs[1])
	plt.plot(m.t[:-1],var_pg/var_pg[0],'-',color='k',alpha=0.5)
	plt.xlabel('T [s]')
	plt.ylabel('Normed Transition Var')
	plt.gca().set_ylim([0,1])
	plt.subplot(gs[0])
	plt.plot(m.t[:-1],np.mean(v_explore,axis=1),color='k')
	plt.xlabel('T [s]')
	plt.ylabel('V explore')
	
	#~ plt.savefig('../../figs/'+fname,bbox_inches='tight')

def parse_input():
	script_help = """ figures.py help
 Sintax:
 figures.py [option flag] [option value]
 
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
 """
	options =  {'bounds_vs_cost':False,'rt_fit':False,'value_and_bounds_sketch':False,
				'confidence_sketch':False,'decision_rt_sketch':False,'bounds_vs_T_n_dt_sketch':False,
				'prior_sketch':False,'vexplore_drop_sketch':False,'show':False}
	keys = options.keys()
	for i,arg in enumerate(sys.argv[1:]):
		if arg=='-h' or arg=='--help':
			print script_help
			sys.exit()
		elif arg=='--all':
			for key in keys:
				if key!='show':
					options[key] = True
		elif arg[2:] in keys:
			options[arg[2:]] = True
		else:
			raise RuntimeError("Unknown option: {opt} encountered in position {pos}. Refer to the help to see the list of options".format(opt=arg,pos=i+1))
	return options

if __name__=="__main__":
	opts = parse_input()
	if opts['bounds_vs_cost']:
		bounds_vs_cost()
	if opts['rt_fit']:
		rt_fit()
	if opts['value_and_bounds_sketch']:
		value_and_bounds_sketch()
	if opts['confidence_sketch']:
		confidence_sketch()
	if opts['decision_rt_sketch']:
		decision_rt_sketch()
	if opts['bounds_vs_T_n_dt_sketch']:
		bounds_vs_T_n_dt_sketch()
	if opts['prior_sketch']:
		prior_sketch()
	if opts['vexplore_drop_sketch']:
		vexplore_drop_sketch()
	
	if opts['show']:
		plt.show(True)
