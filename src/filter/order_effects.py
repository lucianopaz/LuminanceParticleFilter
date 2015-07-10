#!/usr/bin/python
#-*- coding: UTF-8 -*-

from __future__ import division
import math, numpy, random, scipy, copy, sys
import matplotlib as mt
from matplotlib import pyplot as plt
import particle_filter as pf

""" Reproduction of Abbott & Griffiths 2011 """

ad_block=['d','a','a','d','a','d','d','a','a','d',\
		  'c','a','d','a','d','d','a','b','d','a',\
		  'd','a','a','d','d','a','a','c','a','d',\
		  'a','d','a','a','d','b','d','d','a','d']

bc_block=['c','c','a','c','b','b','c','b','b','c',\
		  'b','c','b','b','c','c','d','b','b','c',\
		  'c','c','b','b','c','b','b','b','c','b',\
		  'b','c','a','c','c','b','b','d','c','c']

def get_obs_from_ev(condition):
	if condition=='generative':
		evidence = ad_block+bc_block
	elif condition=='preventative':
		evidence=bc_block+ad_block
	return [[1 if e in ['a','b'] else 0, 1 if e in ['a','c'] else 0] for e in evidence]

def prior():
	return [random.uniform(0,1),random.uniform(-1,1)]

class BetaTrans(pf.TransitionProbDistribution):
	def __init__(self,l=10000.):
		self.l = l
	def rng(self,value):
		return numpy.array([numpy.sign(v)*random.betavariate(self.l*abs(v) + 1, self.l*(1-abs(v)) + 1) for v in value])

def loglike(obs,val):
	if val[1]>=0: # Generative model
		if obs[0]==1 and obs[1]==1:
			ret = val[0] + val[1] - val[0]*val[1]
		elif obs[0]==1 and obs[1]==0:
			ret = 1-(val[0] + val[1] - val[0]*val[1])
		elif obs[0]==0 and obs[1]==1:
			ret = val[0]
		else:
			ret = 1-val[0]
	else: # Preventative model
		if obs[0]==1 and obs[1]==1:
			ret = val[0] * (1 + val[1])
		elif obs[0]==1 and obs[1]==0:
			ret = 1-(val[0] * (1 + val[1]))
		elif obs[0]==0 and obs[1]==1:
			ret = val[0]
		else:
			ret = 1-val[0]
	try:
		ret = math.log(ret)
	except ValueError:
		ret = -float("inf")
	return ret

def experiment(cause_prob,s0,s1):
	if random.random()<=cause_prob:
		cause = 1
		if s1>=0:
			if random.random()<=(s0+s1-s0*s1):
				effect = 1
			else:
				effect = 0
		else:
			if random.random()<=(s0*(1.+s1)):
				effect = 1
			else:
				effect = 0
	else:
		cause = 0
		if random.random()<=s0:
			effect = 1
		else:
			effect = 0
	return [cause,effect]

def get_obs_from_experiment(N_obs,cause_prob,s0,s1):
	return [experiment(cause_prob,s0,s1) for _ in range(N_obs)]

def metropolis_transition(value):
	upper_bound = numpy.array([1,1])
	lower_bound = numpy.array([0,-1])
	ret = random.gauss(value,0.01)
	for i,r in enumerate(ret):
		if r<lower_bound[i]:
			r = lower_bound[i]
		elif r>upper_bound[i]:
			r = upper_bound[i]
		ret[i] = r
	#~ for i,r in enumerate(ret):
		#~ out_of_bounds = r<lower_bound[i] or r>upper_bound[i]
		#~ while out_of_bounds:
			#~ if r<lower_bound[i]:
				#~ r = 2*lower_bound[i]-r
			#~ if r>upper_bound[i]:
				#~ r = 2*upper_bound[i]-r
		#~ ret[i] = r
	return ret

def smooth_kernel(x,value):
	std = 0.05
	u = (numpy.array(x)-value)/std
	return (1/(math.sqrt(2*math.pi)*0.1))*numpy.exp(-u*u/2)

def main(N,R=500,model=2,l=10000,condition='generative',s0=0.5,s1=0.5,pcause=0.5,simlen=80):
	if condition in ['generative','preventative']:
		observations = get_obs_from_ev(condition)
	elif condition=='new':
		observations = get_obs_from_experiment(N_obs=simlen,cause_prob=pcause,s0=s0,s1=s1)
	else:
		raise ValueError("Unknown condition = %s"%condition)
	transition = BetaTrans(l=l)
	mh_ptrans = pf.TransitionProbDistribution(metropolis_transition)
	if model==1:
		filt = pf.ParticleFilter(N,loglike,prior,transition,pf.ResamplingDistribution)
	else:
		filt = pf.ParticleFilter(N,loglike,prior,transition,pf.MultinomialResamplingDistribution)
	
	mean_estimates = numpy.zeros((R,2))
	std_estimates = numpy.zeros((R,2))
	for r in range(R):
		if r==0:
			filt.remember_markov_chain = True
		else:
			filt.remember_markov_chain = False
		filt.reset()
		for trial,observation in enumerate(observations):
			filt.transition(observation)
			if model==2:
				filt.resample()
			else:
				ESS = filt.effective_sample_size()
				if ESS>=0.1*N:
					filt.resample()
					if model==4:
						filt.metropolis_hastings(observation,iterations=10,ptrans=mh_ptrans)
			if filt.remember_markov_chain:
				filt.particle_history.append(filt.particle_set.copy())
		if filt.remember_markov_chain:
			sample_trajectory = filt.particle_history
		mean_estimates[r,:] = filt.particle_set.mean()
		std_estimates[r,:] = filt.particle_set.std()
	plt.figure()
	plt.subplot(121)
	plt.plot(range(R),mean_estimates[:,0],'b')
	plt.fill_between(range(R),mean_estimates[:,0]+std_estimates[:,0],mean_estimates[:,0]-std_estimates[:,0],alpha=0.4,facecolor='b')
	
	plt.subplot(122)
	plt.plot(range(R),mean_estimates[:,1],'r')
	plt.fill_between(range(R),mean_estimates[:,1]+std_estimates[:,1],mean_estimates[:,1]-std_estimates[:,1],alpha=0.4,facecolor='r')
	
	#~ x = numpy.tile(numpy.arange(0.,float(len(observations))),(N,1))
	x = numpy.tile(numpy.arange(0,len(observations)+1).reshape((len(observations)+1,1)),(1,N))
	particle_trajectory = numpy.array([s.v_val for s in sample_trajectory])
	s0_trajectory = particle_trajectory[:,:,0].squeeze()
	s1_trajectory = particle_trajectory[:,:,1].squeeze()
	weight_trajectory = numpy.array([s.weights for s in sample_trajectory])
	mean_s0 = numpy.mean(s0_trajectory,axis=1)
	mean_s1 = numpy.mean(s1_trajectory,axis=1)
	std_s0 = numpy.std(s0_trajectory,axis=1)
	std_s1 = numpy.std(s1_trajectory,axis=1)

	plt.figure()
	plt.subplot(121)
	plt.scatter(x.flatten(1),s0_trajectory.flatten(1),1/weight_trajectory.flatten(1),alpha=0.01)
	plt.plot(x[:,0],mean_s0,'b')
	plt.fill_between(x[:,0],mean_s0+std_s0,mean_s0-std_s0,alpha=0.4,facecolor='b',edgecolor='b')
	plt.subplot(122)
	plt.scatter(x.flatten(1),s1_trajectory.flatten(1),1/weight_trajectory.flatten(1),alpha=0.01)
	plt.plot(x[:,0],mean_s1,'b')
	plt.fill_between(x[:,0],mean_s1+std_s1,mean_s1-std_s1,alpha=0.4,facecolor='b',edgecolor='b')
	
	#~ posterior_function = pf.smooth_posterior(filt.particle_set,smooth_kernel,use_weights=False)
	#~ graph_s0 = numpy.linspace(0, 1, num=100)
	#~ graph_s1 = numpy.linspace(-1, 1, num=100)
	#~ posterior_s0 = numpy.zeros_like(graph_s0)
	#~ posterior_s1 = numpy.zeros_like(graph_s1)
	#~ for i in range(len(graph_s0)):
		#~ p = posterior_function(numpy.array([graph_s0[i],graph_s1[i]]))
		#~ posterior_s0[i] = p[0]
		#~ posterior_s1[i] = p[1]
	#~ plt.plot(range(R),mean_estimates)
	#~ plt.plot(range(R),mean_estimates)
	#~ plt.figure()
	#~ plt.subplot(121)
	#~ plt.plot(graph_s0,posterior_s0)
	#~ plt.subplot(122)
	#~ plt.plot(graph_s1,posterior_s1)
	plt.show()

def parse_args():
	""" Parse arguments passed from the command line """
	arg_list = ['N','R','model','l','condition','s0','s1','pcause','simlen']
	options = {'N':100,'R':500,'model':2,'l':10000.,'condition':'generative','s0':0.5,'s1':0.5,'pcause':0.5,'simlen':80}
	kwarg_flag = False
	skip_arg = True
	arg_n = 0
	for c,arg in enumerate(sys.argv):
		if skip_arg:
			skip_arg = False
			continue
		if not kwarg_flag and arg.startswith('-'):
			kwarg_flag = True
		elif not kwarg_flag:
			if c<4:
				options[arg_list[arg_n]] = int(arg)
			elif c==4:
				options[arg_list[arg_n]] = float(arg)
			elif c==5:
				if arg not in ['generative','preventative','new']:
					raise ValueError("Supplied condition must be either 'generative', 'preventative' or 'new'. User supplied '%s' instead" % (arg))
				options[arg_list[arg_n]] = arg
			elif c<9:
				options[arg_list[arg_n]] = float(arg)
			elif c==9:
				options[arg_list[arg_n]] = int(arg)
			else:
				raise Exception("Unknown fifth option supplied '%s'" %s)
			arg_n+=1
		if kwarg_flag:
			skip_arg = True
			key = arg[1:]
			if key in ['N','R','model','simlen']:
				options[key] = int(sys.argv[c+1])
			elif key in ['l','s0','s1','pcause']:
				options[key] = float(sys.argv[c+1])
			elif key=='condition':
				if sys.argv[c+1] not in ['generative','preventative','new']:
					raise ValueError("Supplied condition must be either 'generative', 'preventative' or 'new'. User supplied '%s' instead" % (sys.argv[c+1]))
				options[key] = sys.argv[c+1]
			elif key in ['h','-help']:
				display_help()
			else:
				raise Exception("Unknown option '%s' supplied" %s)
	return options

def display_help():
	h = """ order_effects.py help
 Sintax:
 order_effects.py [optional arguments]
 
 order_effects.py -h [or --help] displays help
 
 Optional arguments are:
 'N': the number of particles in the particle filter [default 100]
 'R': the number of independent repetitions the filtering method is applied [default 500]
 'model': an integer that indicates the filter model. 1 = never resample, 2 = always resample, 3 = resample when ESS>0.1*N, 4 = same as 3 but after resampling apply metropolis hastings 10 times. [default 2]
 'l': the beta transition probability parameter (high l mean small deviations after transitions) [default 10000]
 'condition': either 'generative', 'preventative' or 'new'. The first two use the same stimuli used in Abbott & Griffiths 2011. The 'new' indicates the stimuli is sampled from the model. [default 'generative']
 's0': only used when condition is 'new'. Is the probability that the effect is produced by the background noise. [default 0.5]
 's1': only used when condition is 'new'. Is the probability that the effect is produced by the cause. Negative values indicate preventative causal effects. [default 0.5]
 'pcause': only used when condition is 'new'. Is the probability that the cause is present. [default 0.5].
 'simlen': only used when condition is 'new'. Is the number of observations that are fed into the filter. [default 80].
 
 Argument can be supplied as positional arguments or as -key value pairs.
 
 Example:
 python order_effects.py 100 -R 3 -model 4 """
	print h
	exit()

if __name__=="__main__":
	options = parse_args()
	print options
	main(**options)
