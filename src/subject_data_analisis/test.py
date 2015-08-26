%pylab
import data_io as io
import kernels as ke
import perfect_inference as pe

figs = []
# data_dir = '/Users/luciano/Facultad/datos'
data_dir = '/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles'
subjects = io.unique_subjects(data_dir)
ms = io.merge_subjects(subjects)
dat,t,d = ms.load_data()
inds = dat[:,1]<1000
dat = dat[inds]
t = np.mean(t[inds],axis=2)
d = np.mean(d[inds],axis=2)
rt_ind = np.floor(dat[:,1]/40)
fluctuations = np.transpose(np.array([t.T-dat[:,0],d.T-50]),(2,0,1))

cumt = ke.center_around(np.cumsum(t,axis=1),rt_ind)
cumd = ke.center_around(np.cumsum(d,axis=1),rt_ind)
cumdiff = cumt-cumd

rdk,rck,rdk_std,rck_std = ke.kernels(fluctuations,1-dat[:,2],dat[:,3]-1,locked_on_onset=False,RT_ind=rt_ind)
centered_ind = np.arange(cumt.shape[1],dtype=float)
rT = (centered_ind-t.shape[1]+1)*40

figs.append(figure(figsize=(13,10)))
plot(rT*1e-3,rdk[0],color='b')
plot(rT*1e-3,rdk[1],color='r')
fill_between(rT*1e-3,rdk[0]-rdk_std[0],rdk[0]+rdk_std[0],color='b',alpha=0.3,edgecolor=None)
fill_between(rT*1e-3,rdk[1]-rdk_std[1],rdk[1]+rdk_std[1],color='r',alpha=0.3,edgecolor=None)
xlabel('T locked on RT [s]')
ylabel('Decision kernel')

figs.append(figure(figsize=(13,10)))
bins = np.linspace(min(np.nanmin(cumt),np.nanmin(cumd)),max(np.nanmax(cumt),np.nanmax(cumd)),501)
bins2 = np.linspace(np.nanmin(cumdiff),np.nanmax(cumdiff),501)
for i in range(10,16):
	label = "T = %1.0f ms"%(rT[i])
	ht,_ = histogram(cumt[~np.isnan(cumt[:,i]),i],bins)
	hd,_ = histogram(cumd[~np.isnan(cumd[:,i]),i],bins)
	hdiff,_ = histogram(cumdiff[~np.isnan(cumdiff[:,i]),i],bins2)
	subplot(221)
	plot(bins[:-1],ht,label=label)
	xlabel('Target cumulative sum')
	subplot(223)
	plot(bins[:-1],hd,label=label)
	xlabel('Distractor cumulative sum')
	subplot(122)
	plot(bins2[:-1],hdiff,label=label)
	xlabel('Difference of cumulative sums')
	legend()
subplot(221).set_xlim([0,1000])
subplot(223).set_xlim([0,1000])
subplot(122).set_xlim([-100,150])
## Probability of subject performance

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

n_sigmas = 10
mat_loglike = np.nan*np.ones((n_sigmas,t.shape[1]+1))
mat_predicted_hits = np.nan*np.ones((n_sigmas,t.shape[1]+1))
mat_predicted_hits_std = np.nan*np.ones((n_sigmas,t.shape[1]+1))
mat_cloglike = np.nan*np.ones((n_sigmas,t.shape[1]*2+1))
mat_cpredicted_hits = np.nan*np.ones((n_sigmas,t.shape[1]*2+1))
mat_cpredicted_hits_std = np.nan*np.ones((n_sigmas,t.shape[1]*2+1))
max_sigma = 30
sigmas = np.linspace(5,max_sigma,n_sigmas)
for i,true_sigma in enumerate(sigmas):
	print i,true_sigma
	model = pe.KnownVarPerfectInference(model_var_t=true_sigma**2,model_var_d=true_sigma**2,prior_mu_t=50.,prior_mu_d=50.,prior_va_t=15**2,prior_va_d=15**2,threshold=1.6,ISI=40)
	mu_t,mu_d,va_t,va_d = model.passiveInference(t,d)
	mu_diff = mu_t-mu_d
	va_diff = va_t+va_d
	prob_miss = normcdf(0.,mu_diff,va_diff)
	cprob_miss = ke.center_around(normcdf(0.,mu_diff,va_diff),rt_ind+1)# Plus 1 because mu_t, mu_d, va_t and va_d's first element is the prior hyperparameter (i.e. no observations)
	trials = np.sum(~np.isnan(cprob_miss),axis=0)
	obs_likelihood = prob_miss.copy()
	obs_likelihood[dat[:,2]==1] = 1.-prob_miss[dat[:,2]==1]
	mat_loglike[i] = np.nansum(np.log(obs_likelihood),axis=0)
	mat_predicted_hits[i] = np.nanmean(1-prob_miss,axis=0)
	mat_predicted_hits_std[i] = np.sqrt(np.nansum((1-prob_miss)*prob_miss,axis=0))/np.sum(~np.isnan(prob_miss),axis=0)
	obs_clikelihood = cprob_miss.copy()
	obs_clikelihood[dat[:,2]==1] = 1.-cprob_miss[dat[:,2]==1]
	mat_cloglike[i] = np.nansum(np.log(obs_clikelihood),axis=0)
	mat_cpredicted_hits[i] = np.nanmean(1-cprob_miss,axis=0)
	mat_cpredicted_hits_std[i] = np.sqrt(np.nansum((1-cprob_miss)*cprob_miss,axis=0))/np.sum(~np.isnan(cprob_miss),axis=0)

colors = plt.get_cmap('jet')

figs.append(figure(figsize=(13,10)))
subplot(321)
gca().set_color_cycle([colors(v) for v in linspace(1,0,mat_cloglike.shape[0])])
plot(np.linspace(-1,1,mat_cloglike.shape[1]),mat_cloglike.T)
ylabel('Log Posterior')
subplot(323)
gca().set_color_cycle([colors(v) for v in linspace(1,0,mat_cloglike.shape[0])])
plot(np.linspace(-1,1,mat_cloglike.shape[1]),trials,'--k')
ylabel('Valid data samples')
subplot(325)
gca().set_color_cycle([colors(v) for v in linspace(1,0,mat_cloglike.shape[0])])
plot(np.linspace(-1,1,mat_cloglike.shape[1]),(mat_cloglike/trials).T)
gca().set_ylim([-0.7,-0.4])
ylabel('Log Posterior averaged on valid data samples')
xlabel('T [s]')
subplot(122)
gca().set_color_cycle([colors(v) for v in linspace(1,0,mat_cloglike.shape[0])])
plot([-1,1],np.mean(dat[:,2])*np.ones(2),'--k',label='Subject')
for i,(h,hs) in enumerate(zip(mat_cpredicted_hits,mat_cpredicted_hits_std)):
	errorbar(np.linspace(-1,1,h.shape[0]), h, yerr=hs,label="$\sigma=%1.2f$"%(sigmas[i]))
legend(loc=0)
plt.gca().set_ylim([0,1])
xlabel('T [s]')
ylabel('Predicted performance')
suptitle('Centered around decision time')

figs.append(figure(figsize=(13,10)))
subplot(121)
gca().set_color_cycle([colors(v) for v in linspace(1,0,mat_loglike.shape[0])])
plot(np.linspace(0,1,mat_loglike.shape[1]),mat_loglike.T)
xlabel('T [s]')
ylabel('Log Posterior')
subplot(122)
gca().set_color_cycle([colors(v) for v in linspace(1,0,mat_loglike.shape[0])])
plot([0,1],np.mean(dat[:,2])*np.ones(2),'--k',label='Subject')
for i,(h,hs) in enumerate(zip(mat_predicted_hits,mat_predicted_hits_std)):
	errorbar(np.linspace(0,1,h.shape[0]), h, yerr=hs,label="$\sigma=%1.2f$"%(sigmas[i]))
plt.gca().set_ylim([0,1])
legend(loc=0)
xlabel('T [s]')
ylabel('Performance')
suptitle('Locked on stim onset')

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('../../figs/Subject_posterior.pdf') as pdf:
	for fig in figs:
		pdf.savefig(fig)
