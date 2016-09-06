from __future__ import division

import numpy as np
import data_io_cognition as io
import cost_time as ct
import fits_cognition as fits
from matplotlib import pyplot as plt
from utils import normpdf
import scipy.signal

subjects = io.filter_subjects_list(io.unique_subject_sessions(fits.raw_data_dir),'all_sessions_by_experiment')
#~ subjects = io.filter_subjects_list(subjects,'experiment_2AFC')
#~ subjects = io.filter_subjects_list(subjects,'experiment_Auditivo')
subjects = io.filter_subjects_list(subjects,'experiment_Luminancia')
s = subjects[0]
fitter = fits.Fitter(s)
fitter.set_fixed_parameters({'high_confidence_threshold':1.08,'confidence_map_slope':8.})
fitter.set_start_point({'internal_var':0.001})
fitter.set_bounds()
fitter.fixed_parameters,fitter.fitted_parameters,start_point,bounds = fitter.sanitize_parameters_x0_bounds()
parameters = fitter.get_parameters_dict(start_point)
#~ print parameters
m = fitter.dp
m.set_cost(parameters['cost'])
if not fitter.experiment is 'Luminancia':
	m.set_internal_var(parameters['internal_var'])
xub,xlb,v,v_explore,v1,v2 = m.xbounds(return_values=True)
first_passage_pdf = np.array(m.rt(0.,bounds=(xub,xlb)))

#~ print m.prior_mu_var
#~ print np.array(1.)/np.array(0.)
#~ plt.figure()
#~ plt.subplot(121)
#~ plt.plot(m.t,m.bounds.T)
#~ plt.subplot(122)
#~ plt.plot(m.t,np.array([xub,xlb]).T)
#~ print xub,xlb
#~ 
#~ plt.figure()
#~ plt.subplot(221)
#~ plt.imshow(v.T,aspect="auto",interpolation='none',extent=[m.t[0],m.t[-1],0,1],origin='lower')
#~ plt.colorbar()
#~ plt.subplot(223)
#~ plt.imshow(v.T,aspect="auto",interpolation='none',extent=[m.t[0],m.t[-1],0,1],origin='lower')
#~ plt.colorbar()
#~ plt.subplot(122)
#~ plt.plot(v1)
#~ plt.plot(v2)
#~ plt.show(True)


confidence_value = fitter.high_confidence_mapping(parameters['high_confidence_threshold'],parameters['confidence_map_slope'])
plt.figure()
plt.subplot(121)
plt.plot(m.t,confidence_value.T)
plt.subplot(122)
plt.plot(m.t,first_passage_pdf.T)

conv_confidence_matrix,confidence_matrix = fitter.confidence_mapping_pdf_matrix(first_passage_pdf,parameters,return_unconvoluted_matrix=True)
conv_t = np.arange(0,conv_confidence_matrix.shape[2],dtype=np.float)*m.dt

plt.figure()
ax1 = plt.subplot(221)
plt.imshow(confidence_matrix[0],aspect="auto",interpolation='none',extent=[0,confidence_matrix.shape[2]-1,0,1],origin='lower')
plt.subplot(223,sharex=ax1,sharey=ax1)
plt.imshow(confidence_matrix[1],aspect="auto",interpolation='none',extent=[0,confidence_matrix.shape[2]-1,0,1],origin='lower')
ax2 = plt.subplot(222,sharey=ax1)
plt.imshow(conv_confidence_matrix[0],aspect="auto",interpolation='none',extent=[0,conv_confidence_matrix.shape[2]-1,0,1],origin='lower')
plt.colorbar()
ax2 = plt.subplot(224,sharex=ax2,sharey=ax2)
plt.imshow(conv_confidence_matrix[1],aspect="auto",interpolation='none',extent=[0,conv_confidence_matrix.shape[2]-1,0,1],origin='lower')
plt.colorbar()

#~ plt.figure()
#~ plt.plot(np.sum(confidence_matrix,axis=0))
#~ plt.plot(np.sum(conv_confidence_matrix,axis=0))
#~ print np.sum(confidence_matrix),np.sum(conv_confidence_matrix),np.sum(first_passage_pdf[0])

#~ plt.figure()
#~ conf_array = np.linspace(0,1,conv_confidence_matrix.shape[0])
#~ conf_prob = np.sum(conv_confidence_matrix,axis=1)*m.dt
#~ mean_conf = np.sum(conv_confidence_matrix.T*conf_array,axis=1)/np.sum(conv_confidence_matrix,axis=0)
#~ var_conf1 = np.sum(conv_confidence_matrix.T*conf_array**2,axis=1)/np.sum(conv_confidence_matrix,axis=0)-mean_conf**2
#~ var_conf = np.sum(conv_confidence_matrix.T*(np.tile(conf_array,(m.nT,1)).T-mean_conf).T**2,axis=1)/np.sum(conv_confidence_matrix,axis=0)
#~ print np.max(np.abs(var_conf1[var_conf1>=0.]-var_conf[var_conf>=0.]))
#~ plt.subplot(121)
#~ plt.plot(conf_array,np.sum(conv_confidence_matrix,axis=1)*m.dt)
#~ ax = plt.subplot(222)
#~ plt.fill_between(m.t,mean_conf+np.sqrt(var_conf),mean_conf-np.sqrt(var_conf),color='b',alpha=0.6)
#~ plt.plot(m.t,mean_conf,linewidth=2,color='b')
#~ plt.subplot(224,sharex=ax)
#~ plt.plot(m.t,np.sqrt(var_conf))

plt.figure()
ax = plt.subplot(121)
plt.imshow(conv_confidence_matrix[0],aspect="auto",interpolation='none',extent=[conv_t[0],conv_t[-1]-1,0,1],origin='lower')
plt.colorbar()
plt.subplot(122,sharex=ax,sharey=ax)
plt.imshow(conv_confidence_matrix[1],aspect="auto",interpolation='none',extent=[conv_t[0],conv_t[-1]-1,0,1],origin='lower')
plt.colorbar()
print 0.2,0.5,fits.rt_confidence_likelihood(conv_t,conv_confidence_matrix[0],0.2,0.5)
print 0.2,0.7,fits.rt_confidence_likelihood(conv_t,conv_confidence_matrix[0],0.2,0.7)
print 0.4,0.7,fits.rt_confidence_likelihood(conv_t,conv_confidence_matrix[0],0.4,0.7)
print 0.5,0.02,fits.rt_confidence_likelihood(conv_t,conv_confidence_matrix[0],0.5,0.02)
print 0.65,0.3,fits.rt_confidence_likelihood(conv_t,conv_confidence_matrix[0],0.65,0.3)
print 0.35,0.91,fits.rt_confidence_likelihood(conv_t,conv_confidence_matrix[0],0.35,0.91)

#~ plt.figure()
#~ ax1 = plt.subplot(121)
#~ plt.imshow(confidence_matrix,aspect="auto",interpolation='none',extent=[0,confidence_matrix.shape[1]-1,0,1],origin='lower')
#~ plt.plot(confidence_value[0],'or')
#~ plt.subplot(222,sharex=ax1)
#~ plt.plot(np.sum(confidence_matrix,axis=0),'b')
#~ plt.plot(first_passage_pdf[0],'r')
#~ plt.subplot(224,sharex=ax1)
#~ plt.plot(first_passage_pdf[0]-np.sum(confidence_matrix,axis=0))


#~ miss_confidence,hit_confidence,conv_x = fitter.confidence_probability_time_matrix(first_passage_pdf,parameters,return_convx=True,confidence_partition=50.)
#~ plt.figure()
#~ plt.subplot(211)
#~ plt.imshow(hit_confidence,aspect="auto",interpolation='none',extent=[conv_x[0],conv_x[-1],0,1],origin='lower')
#~ plt.subplot(212)
#~ plt.imshow(miss_confidence,aspect="auto",interpolation='none',extent=[conv_x[0],conv_x[-1],0,1],origin='lower')

#~ print hit_confidence.T

#~ plt.figure()
#~ plt.subplot(211)
#~ plt.plot(conv_x,hit_confidence.T)
#~ plt.subplot(212)
#~ plt.plot(conv_x,miss_confidence.T)
#~ 
#~ print np.sum((hit_confidence+miss_confidence)*_dt/(miss_confidence.shape[0]))
plt.show(True)
