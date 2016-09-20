from __future__ import division

import numpy as np
import data_io_cognition as io
import cost_time as ct
import fits_cognition as fits
from fits_cognition import Fitter
import matplotlib as mt
mt.use("Qt4Agg") # This program works with Qt only
from matplotlib import pyplot as plt
from matplotlib import widgets
import os,re

### control panel ###
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtCore import Qt

class Sweeper:
	def __init__(self,output_fname='fits_cognition/manual_fits.dat'):
		if os.path.exists(output_fname) and os.path.isfile(output_fname):
			output_file = open(output_fname,'r+')
			lines = output_file.readlines()
			keys = [k.strip(' \r\n') for k in lines[::3]]
		subjects = io.filter_subjects_list(io.filter_subjects_list(io.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment'),'experiment_Luminancia')
		self.subjects = [s for s in io.merge_subjectSessions(subjects,merge='sessions') if not s.get_key() in keys]
		#~ subjects = io.filter_subjects_list(io.filter_subjects_list(io.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment'),'experiment_Auditivo')
		#~ self.subjects = [s for s in subjects if not s.get_key() in keys]
		self.n_subjects = len(self.subjects)
		self.current_subject = 0
		if self.n_subjects==0:
			print 'All subjects have been fitted. If you wish to manually fit the subjects again you must change the name of the file {0} or remove its content'.format(output_fname)
		self.subjects = iter(self.subjects)
		
		self.prepare_next_subject()
		self.prepare_fitter()
		self.output_file = open(output_fname,'a+')
		self.aux_open = False
	
	def start(self):
		self.create_fig()
		self.init_plot()
		self.init_widgets()
		plt.show(True)
	
	def end(self):
		try:
			self.close()
		except:
			pass
		print 'Completed all fits'
	
	def prepare_next_subject(self):
		try:
			self.subjectSession = self.subjects.next()
			self.current_subject+= 1
		except StopIteration:
			self.end()
	
	def prepare_fitter(self):
		try:
			fname = 'fits_cognition/{experiment}_fit_confidence_only_subject_{name}_session_{session}_cma.pkl'.format(
				experiment=self.subjectSession.experiment,name=self.subjectSession.name,session=self.subjectSession.get_session())
			self.fitter = fits.load_Fitter_from_file(fname)
			self.parameters = self.fitter.get_parameters_dict_from_fit_output(self.fitter._fit_output)
			self.fitter.set_fixed_parameters({})
		except:
			self.fitter = fits.Fitter(self.subjectSession,decisionPolicyKwArgs={'n':201})
			self.fitter.set_fixed_parameters({})
			self.parameters = self.fitter.default_start_point()
		#~ print self.fitter.default_start_point(True)
		#~ self.fitter.set_start_point({'cost':0.,'internal_var':7000.,'dead_time':0.5,'dead_time_sigma':0.08,'phase_out_prob':0.05})
		#~ self.fitter.set_bounds()
		#~ self.fitter.fixed_parameters,self.fitter.fitted_parameters,start_point = self.fitter.sanitize_parameters_x0_bounds()[:3]
		#~ self.parameters = self.fitter.get_parameters_dict_from_array(start_point)
		self.set_parameters()
	
	def set_parameters(self,d={}):
		self.parameters.update(d)
		self.fitter.dp.set_cost(self.parameters['cost'])
		self.fitter.dp.set_internal_var(self.parameters['internal_var'])
	
	def compute_bounds(self):
		self.xub,self.xlb = self.fitter.dp.xbounds()
		self.first_passage_pdf = None
		self._gs = {}
		for index,drift in enumerate(self.fitter.mu):
			gs = np.array(self.fitter.dp.rt(drift,bounds=(self.xub,self.xlb)))
			if self.first_passage_pdf is None:
				self.first_passage_pdf = gs*self.fitter.mu_prob[index]
			else:
				self.first_passage_pdf+= gs*self.fitter.mu_prob[index]
			self._gs[drift] = gs
	
	def compute_confidence_mapping(self):
		self.confidence_mapping = self.fitter.high_confidence_mapping(self.parameters['high_confidence_threshold'],self.parameters['confidence_map_slope'])
	
	def compute_predition(self):
		random_rt_likelihood = 0.5*self.parameters['phase_out_prob']/(self.fitter.max_RT-self.fitter.min_RT)/self.fitter.confidence_partition
		self.distribution = None
		full_nLL = 0.
		confidence_only_nLL = 0.
		full_confidence_nLL = 0.
		for index,drift in enumerate(self.fitter.mu):
			gs = self._gs[drift]
			dec_rt_confidence_likelihood = self.fitter.confidence_mapping_pdf_matrix(gs,self.parameters,mapped_confidences=self.confidence_mapping)
			rt_confidence_likelihood = np.sum(dec_rt_confidence_likelihood,axis=0)
			rt_likelihood = np.sum(dec_rt_confidence_likelihood,axis=1)
			if self.distribution is None:
				self.distribution = dec_rt_confidence_likelihood*self.fitter.mu_prob[index]
			else:
				self.distribution+= dec_rt_confidence_likelihood*self.fitter.mu_prob[index]
			t = np.arange(0,dec_rt_confidence_likelihood.shape[-1],dtype=np.float)*self.fitter.dp.dt
			indeces = self.fitter.mu_indeces==index
			for rt,perf,conf in zip(self.fitter.rt[indeces],self.fitter.performance[indeces],self.fitter.confidence[indeces]):
				temp = fits.rt_likelihood(t,rt_likelihood[1-int(perf)],rt)*(1-self.parameters['phase_out_prob'])+random_rt_likelihood*self.fitter.confidence_partition
				#~ print temp
				full_nLL-=np.log(fits.rt_likelihood(t,rt_likelihood[1-int(perf)],rt)*(1-self.parameters['phase_out_prob'])+random_rt_likelihood*self.fitter.confidence_partition)
				confidence_only_nLL-=np.log(fits.confidence_likelihood(np.sum(dec_rt_confidence_likelihood,axis=2)[1-int(perf)],conf)*(1-self.parameters['phase_out_prob'])+random_rt_likelihood*(self.fitter.max_RT-self.fitter.min_RT))
				full_confidence_nLL-=np.log(fits.rt_confidence_likelihood(t,dec_rt_confidence_likelihood[1-int(perf)],rt,conf)*(1-self.parameters['phase_out_prob'])+random_rt_likelihood)
		random_rt_likelihood = random_rt_likelihood*np.ones_like(t)
		random_rt_likelihood[np.logical_or(t<self.fitter.min_RT,t>self.fitter.max_RT)] = 0.
		self.distribution = self.distribution*(1-self.parameters['phase_out_prob'])+random_rt_likelihood
		self.distribution/=(np.sum(self.distribution)*self.fitter.dp.dt)
		self.model_rt = np.sum(self.distribution,axis=1)
		self.model_conf = np.sum(self.distribution,axis=2)*self.fitter.dp.dt
		self.model_t = t
		self.full_nLL = full_nLL
		self.confidence_only_nLL = confidence_only_nLL
		self.full_confidence_nLL = full_confidence_nLL
	
	def get_subject_data(self):
		self._max_RT = np.ceil(self.fitter.max_RT) if self.fitter.time_units=='seconds' else np.ceil(self.fitter.max_RT/1000)*1000
		edges = np.linspace(0,self._max_RT,51)
		rt_edges = edges
		self.rt_centers = np.array([0.5*(e1+e0) for e1,e0 in zip(rt_edges[1:],rt_edges[:-1])])
		c_edges = np.linspace(0,1,self.fitter.confidence_partition+1)
		self.c_centers = np.array([0.5*(e1+e0) for e1,e0 in zip(c_edges[1:],c_edges[:-1])])
		dt = rt_edges[1]-rt_edges[0]
		dc = c_edges[1]-c_edges[0]
		hit = self.fitter.performance==1
		miss = np.logical_not(hit)
		subject_hit_histogram2d = np.histogram2d(self.fitter.rt[hit], self.fitter.confidence[hit], bins=[rt_edges,c_edges])[0].astype(np.float).T
		subject_miss_histogram2d = np.histogram2d(self.fitter.rt[miss], self.fitter.confidence[miss], bins=[rt_edges,c_edges])[0].astype(np.float).T
		norm = (np.sum(subject_hit_histogram2d)+np.sum(subject_miss_histogram2d))*dt
		
		self.subject_rt = np.array([np.sum(subject_hit_histogram2d,axis=0),
									np.sum(subject_miss_histogram2d,axis=0)])/norm
		self.subject_conf = np.array([np.sum(subject_hit_histogram2d,axis=1),
									  np.sum(subject_miss_histogram2d,axis=1)])*dt/norm
	
	def create_fig(self):
		self.fig = plt.figure(figsize=(12,9))
		self.rt_ax = plt.subplot(121)
		self.conf_ax = plt.subplot(122)
	
	def create_auxiliary_plots(self,event=None):
		if not self.aux_open:
			self.aux_fig = plt.figure(figsize=(8,12))
			self.aux_fig.canvas.mpl_connect('close_event', self.closed_auxiliary_plots)
			self.fig.canvas.mpl_connect('close_event', self.close_auxiliary_plots)
			self.aux_open = True
		else:
			self.aux_fig.clear()
		self.plot_auxiliary_plots()
	
	def init_widgets(self):
		root = self.fig.canvas.manager.window
		panel = QtGui.QWidget()
		hbox = QtGui.QHBoxLayout(panel)
		self.cost_textbox = QtGui.QLineEdit(str(self.parameters['cost']),parent = panel)
		self.internal_var_textbox = QtGui.QLineEdit(str(self.parameters['internal_var']),parent = panel)
		self.dead_time_textbox = QtGui.QLineEdit(str(self.parameters['dead_time']),parent = panel)
		self.dead_time_sigma_textbox = QtGui.QLineEdit(str(self.parameters['dead_time_sigma']),parent = panel)
		self.phase_out_prob_textbox = QtGui.QLineEdit(str(self.parameters['phase_out_prob']),parent = panel)
		self.high_confidence_threshold_textbox = QtGui.QLineEdit(str(self.parameters['high_confidence_threshold']),parent = panel)
		self.confidence_map_slope_textbox = QtGui.QLineEdit(str(self.parameters['confidence_map_slope']),parent = panel)
		self.cost_textbox.returnPressed.connect(self.update_cost)
		self.internal_var_textbox.returnPressed.connect(self.update_internal_var)
		self.dead_time_textbox.returnPressed.connect(self.update_dead_time)
		self.dead_time_sigma_textbox.returnPressed.connect(self.update_dead_time_sigma)
		self.phase_out_prob_textbox.returnPressed.connect(self.update_phase_out_prob)
		self.high_confidence_threshold_textbox.returnPressed.connect(self.update_high_confidence_threshold)
		self.confidence_map_slope_textbox.returnPressed.connect(self.update_confidence_map_slope)
		hbox.addWidget(self.cost_textbox)
		hbox.addWidget(self.internal_var_textbox)
		hbox.addWidget(self.dead_time_textbox)
		hbox.addWidget(self.dead_time_sigma_textbox)
		hbox.addWidget(self.phase_out_prob_textbox)
		hbox.addWidget(self.high_confidence_threshold_textbox)
		hbox.addWidget(self.confidence_map_slope_textbox)
		panel.setLayout(hbox)
		
		dock = QtGui.QDockWidget("control: cost, sigma, tau_c, sigma_c, p_po, C_H, conf_map_slope", root)
		root.addDockWidget(Qt.BottomDockWidgetArea, dock)
		dock.setWidget(panel)
		
		self.button_ax = plt.axes([0.75, 0.92, 0.20, 0.075],frameon=True)
		self.continue_button = widgets.Button(self.button_ax, 'Save and continue')
		self.continue_button.on_clicked(self.save_and_continue)
		
		self.button_ax2 = plt.axes([0.05, 0.92, 0.20, 0.075],frameon=True)
		self.aux_plots = widgets.Button(self.button_ax2, 'Aux plots')
		self.aux_plots.on_clicked(self.create_auxiliary_plots)
	
	def init_plot(self):
		self.get_subject_data()
		self.compute_bounds()
		self.compute_confidence_mapping()
		self.compute_predition()
		
		plt.sca(self.rt_ax)
		plt.step(self.rt_centers,self.subject_rt[0],'b',label='Subject hit')
		plt.step(self.rt_centers,-self.subject_rt[1],'r',label='Subject miss')
		self.hit_rt, = plt.plot(self.model_t,self.model_rt[0],'b',label='Model hit',linewidth=2)
		self.miss_rt, = plt.plot(self.model_t,-self.model_rt[1],'r',label='Model hit',linewidth=2)
		self.rt_ax.set_xlim([0,self._max_RT])
		plt.xlabel('RT')
		plt.ylabel('Prob density')
		plt.legend(loc='best', fancybox=True, framealpha=0.5)
		
		plt.sca(self.conf_ax)
		plt.step(self.c_centers,self.subject_conf[0],'b',label='Subject hit')
		plt.step(self.c_centers,-self.subject_conf[1],'r',label='Subject miss')
		self.hit_conf, = plt.plot(np.linspace(0,1,self.model_conf.shape[1]),self.model_conf[0],'b',label='Model hit',linewidth=2)
		self.miss_conf, = plt.plot(np.linspace(0,1,self.model_conf.shape[1]),-self.model_conf[1],'r',label='Model hit',linewidth=2)
		plt.xlabel('RT')
		plt.ylabel('Prob density')
		plt.legend(loc='best', fancybox=True, framealpha=0.5)
		
		self.title = plt.suptitle('full_nLL = {full_nLL}\nconfidence_only_nLL = {confidence_only_nLL}\nfull_confidence_nLL = {full_confidence_nLL}\n'.format(
						full_nLL=self.full_nLL,confidence_only_nLL=self.confidence_only_nLL,full_confidence_nLL=self.full_confidence_nLL))
	
	def update_plot(self):
		self.hit_rt.set_xdata(self.model_t)
		self.hit_rt.set_ydata(self.model_rt[0])
		self.miss_rt.set_xdata(self.model_t)
		self.miss_rt.set_ydata(-self.model_rt[1])
		self.hit_conf.set_ydata(self.model_conf[0])
		self.miss_conf.set_ydata(-self.model_conf[1])
		self.title.set_text('full_nLL = {full_nLL}\nconfidence_only_nLL = {confidence_only_nLL}\nfull_confidence_nLL = {full_confidence_nLL}\n'.format(
						full_nLL=self.full_nLL,confidence_only_nLL=self.confidence_only_nLL,full_confidence_nLL=self.full_confidence_nLL))
		self.fig.canvas.draw()
		self.plot_auxiliary_plots()
	
	def plot_auxiliary_plots(self):
		if self.aux_open:
			self.aux_fig.clear()
			plt.figure(self.aux_fig.number)
			ax = self.aux_fig.add_subplot(321)
			ax.plot(self.fitter.dp.t,self.fitter.dp.bounds.T)
			ax.set_ylabel('g bounds')
			ax = self.aux_fig.add_subplot(323)
			ax.plot(self.fitter.dp.t,np.array([self.xub,self.xlb]).T)
			ax.set_ylabel('x bounds')
			ax = self.aux_fig.add_subplot(325)
			ax.plot(self.fitter.dp.t,self.first_passage_pdf.T)
			ax.set_ylabel('FPT pdf')
			ax.set_xlabel('T')
			ax = self.aux_fig.add_subplot(322)
			ax.plot(self.fitter.dp.t,self.fitter.dp.log_odds().T)
			ax.set_ylabel('Log odds')
			ax = self.aux_fig.add_subplot(324)
			ax.plot(self.fitter.dp.t,self.confidence_mapping.T)
			ax.set_xlabel('T')
			ax.set_ylabel('Conf mapping')
			ax = self.aux_fig.add_subplot(326)
			ax.imshow(self.distribution[0],aspect="auto",interpolation='none',origin='lower',
						extent=[self.model_t[0],self.model_t[-1],0,1])
			ax.set_ylabel('C')
			ax.set_xlabel('T')
			self.aux_fig.canvas.draw()
	
	def update_cost(self):
		try:
			self.set_parameters({'cost':float(self.cost_textbox.text())})
		except:
			return
		self.compute_bounds()
		self.compute_confidence_mapping()
		self.compute_predition()
		self.update_plot()
	
	def update_internal_var(self):
		try:
			self.set_parameters({'internal_var':float(self.internal_var_textbox.text())})
		except:
			return
		self.compute_bounds()
		self.compute_confidence_mapping()
		self.compute_predition()
		self.update_plot()
	
	def update_dead_time(self):
		try:
			self.set_parameters({'dead_time':float(self.dead_time_textbox.text())})
		except:
			return
		self.compute_predition()
		self.update_plot()
	
	def update_dead_time_sigma(self):
		try:
			self.set_parameters({'dead_time_sigma':float(self.dead_time_sigma_textbox.text())})
		except:
			return
		self.compute_predition()
		self.update_plot()
	
	def update_phase_out_prob(self):
		try:
			self.set_parameters({'phase_out_prob':float(self.phase_out_prob_textbox.text())})
		except:
			return
		self.compute_predition()
		self.update_plot()
	
	def update_high_confidence_threshold(self):
		try:
			self.set_parameters({'high_confidence_threshold':float(self.high_confidence_threshold_textbox.text())})
		except:
			return
		self.compute_confidence_mapping()
		self.compute_predition()
		self.update_plot()
	
	def update_confidence_map_slope(self):
		try:
			self.set_parameters({'confidence_map_slope':float(self.confidence_map_slope_textbox.text())})
		except:
			return
		self.compute_confidence_mapping()
		self.compute_predition()
		self.update_plot()
	
	def reset_textboxes(self):
		self.cost_textbox.setText(str(self.parameters['cost']))
		self.internal_var_textbox.setText(str(self.parameters['internal_var']))
		self.dead_time_textbox.setText(str(self.parameters['dead_time']))
		self.dead_time_sigma_textbox.setText(str(self.parameters['dead_time_sigma']))
		self.phase_out_prob_textbox.setText(str(self.parameters['phase_out_prob']))
		self.high_confidence_threshold_textbox.setText(str(self.parameters['high_confidence_threshold']))
		self.confidence_map_slope_textbox.setText(str(self.parameters['confidence_map_slope']))
	
	def save_and_continue(self,event=None):
		header = self.subjectSession.get_key()+'\n'
		nLL = 'full_nLL={full_nLL}\tconfidence_only_nLL={confidence_only_nLL}\tfull_confidence_nLL={full_confidence_nLL}\n'.format(
						full_nLL=self.full_nLL,confidence_only_nLL=self.confidence_only_nLL,full_confidence_nLL=self.full_confidence_nLL)
		pars = '{parameters}\n'.format(parameters=self.parameters)
		self.output_file.write(header)
		self.output_file.write(nLL)
		self.output_file.write(pars)
		self.output_file.flush()
		print header,nLL,pars,'Completed {0}/{1} fits'.format(self.current_subject,self.n_subjects)
		
		self.prepare_next_subject()
		self.prepare_fitter()
		self.clean_plot()
		self.init_plot()
		self.reset_textboxes()
		self.close_auxiliary_plots()
	
	def clean_plot(self):
		self.rt_ax.clear()
		self.conf_ax.clear()
	
	def close(self):
		self.output_file.close()
		plt.close(self.fig)
	
	def closed_auxiliary_plots(self,event=None):
		self.aux_open = False
	
	def close_auxiliary_plots(self,event=None):
		if self.aux_open:
			plt.close(self.aux_fig)

if __name__=="__main__":
	sweeper = Sweeper()
	sweeper.start()
