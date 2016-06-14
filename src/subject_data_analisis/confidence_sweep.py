from __future__ import division

import enum, os, sys, math, scipy, pickle, cma, copy
import data_io as io
import cost_time as ct
import numpy as np
from utils import normpdf
import matplotlib as mt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import moving_bounds_fits as mo
from matplotlib.widgets import Slider

class Sweeper:
	subjects = io.unique_subjects(mo.data_dir)
	subjects.append(io.merge_subjects(subjects))
	fit_files = ["fits/inference_fit_full_subject_{0}.pkl".format(s.id) for s in subjects]
	
	def sweep(self):
		for i,(s,fname) in enumerate(zip(self.subjects,self.fit_files)):
			self.sweep_plot(i,s,fname)
			plt.show(True)
	
	def sweep_plot(self,i,s,fname):
		f = open(fname)
		out = pickle.load(f)
		f.close()
		try:
			self.cost = out[0]['cost']
			self.dead_time = out[0]['dead_time']
			self.dead_time_sigma = out[0]['dead_time_sigma']
			self.phase_out_prob = out[0]['phase_out_prob']
		except:
			self.cost = out[0][0]
			self.dead_time = out[0][1]
			self.dead_time_sigma = out[0][2]
			self.phase_out_prob = out[0][3]
		self.dat,t,d = s.load_data()
		rt = self.dat[:,1]*1e-3
		self.max_RT = np.max(rt)
		perf = self.dat[:,2]
		conf = self.dat[:,3]
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
		
		self.mu,self.mu_indeces,count = np.unique((self.dat[:,0]-mo.distractor)/mo.ISI,return_inverse=True,return_counts=True)
		self.mu_prob = count.astype(np.float64)
		self.mu_prob/=np.sum(self.mu_prob)
		mus = np.concatenate((-self.mu[::-1],self.mu))
		counts = np.concatenate((count[::-1].astype(np.float64),count.astype(np.float64)))*0.5
		p = counts/np.sum(counts)
		prior_mu_var = np.sum(p*(mus-np.sum(p*mus))**2)
		self.m = ct.DecisionPolicy(model_var=mo.model_var,prior_mu_var=prior_mu_var,n=101,T=10,dt=mo.ISI,reward=1,penalty=0,iti=1.,tp=0.,store_p=False)
		
		dec_rt,self.dec_gs = mo.decision_rt_distribution(self.cost,self.dead_time,self.dead_time_sigma,self.phase_out_prob,self.m,self.mu,self.mu_prob,self.max_RT,return_gs=True,include_t0=False)
		
		self.fig = plt.figure()
		mxlim = np.ceil(self.max_RT)
		self.axslider = plt.axes([0.6, 0.05, 0.35, 0.03])
		self.slider = Slider(self.axslider, 'High confidence threshold', 0, 2, valinit=0.8)
		self.slider.on_changed(self.update)
		
		self.dec_ax = plt.subplot(121)
		plt.step(xh,hit_rt,label='Subject '+str(s.id)+' hit rt',where='post',color='b')
		plt.step(xh,-miss_rt,label='Subject '+str(s.id)+' miss rt',where='post',color='r')
		plt.plot(self.m.t[1:],dec_rt['full']['all'][0],label='Theoretical hit rt',linewidth=2,color='b')
		plt.plot(self.m.t[1:],-dec_rt['full']['all'][1],label='Theoretical miss rt',linewidth=2,color='r')
		plt.xlim([0,mxlim])
		plt.xlabel('T [s]')
		plt.ylabel('Prob density')
		plt.legend()
		
		self.conf_ax = plt.subplot(122)
		confidence_params = [self.slider.val]
		conf_rt = mo.confidence_rt_distribution(self.dec_gs,self.cost,self.dead_time,self.dead_time_sigma,self.phase_out_prob,self.m,self.mu,self.mu_prob,self.max_RT,confidence_params,include_t0=False)
		params = [self.cost, self.dead_time, self.dead_time_sigma, self.phase_out_prob, self.slider.val]
		self.nLL = mo.full_confidence_merit(params,self.m,self.dat,self.mu,self.mu_indeces)
		plt.step(xh,high_hit_rt+high_miss_rt,label='Subject '+str(s.id)+' high',where='post',color='b')
		plt.step(xh,-(low_hit_rt+low_miss_rt),label='Subject '+str(s.id)+' low',where='post',color='r')
		l1, = plt.plot(self.m.t[1:],np.sum(conf_rt['full']['high'],axis=0),label='Theoretical high',linewidth=2,color='b')
		l2, = plt.plot(self.m.t[1:],-np.sum(conf_rt['full']['low'],axis=0),label='Theoretical low',linewidth=2,color='r')
		self.lines = [l1,l2]
		#~ l11,l12 = plt.plot(self.m.t[1:],conf_rt['full']['high'].T,label='Theoretical high',linewidth=2,color='b')
		#~ l21,l22 = plt.plot(self.m.t[1:],-conf_rt['full']['low'].T,label='Theoretical low',linewidth=2,color='r')
		#~ self.lines = [l11,l12,l21,l22]
		plt.xlim([0,mxlim])
		plt.xlabel('T [s]')
		self.conf_ax.set_title("nLL = {0}".format(self.nLL))
		plt.legend()
		
	
	def update(self,val):
		confidence_params = [self.slider.val]
		conf_rt = mo.confidence_rt_distribution(self.dec_gs,self.cost,self.dead_time,self.dead_time_sigma,self.phase_out_prob,self.m,self.mu,self.mu_prob,self.max_RT,confidence_params,include_t0=False)
		self.lines[0].set_ydata(np.sum(conf_rt['full']['high'],axis=0))
		self.lines[1].set_ydata(-np.sum(conf_rt['full']['low'],axis=0))
		params = [self.cost, self.dead_time, self.dead_time_sigma, self.phase_out_prob, self.slider.val]
		self.nLL = mo.full_confidence_merit(params,self.m,self.dat,self.mu,self.mu_indeces)
		self.conf_ax.set_title("nLL = {0}".format(self.nLL))
		#~ self.lines[0].set_ydata(conf_rt['full']['high'][0])
		#~ self.lines[1].set_ydata(conf_rt['full']['high'][1])
		#~ self.lines[2].set_ydata(-conf_rt['full']['low'][0])
		#~ self.lines[3].set_ydata(-conf_rt['full']['low'][1])

if __name__=="__main__":
	sweeper = Sweeper()
	sweeper.sweep()
