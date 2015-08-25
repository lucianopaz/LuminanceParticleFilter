%pylab
import data_io as io
import kernels as ke
import perfect_inference as pe

data_dir = '/Users/luciano/Facultad/datos'
# data_dir = '/home/luciano/facultad/dropbox_backup_2015_02_03/LuminanceConfidenceKernels/data/controles'
subjects = io.unique_subjects(data_dir)
ms = io.merge_subjects(subjects)
dat,t,d = ms.load_data()
inds = dat[:,1]<1000
dat = dat[inds]
t = np.mean(t[inds],axis=2)
d = np.mean(d[inds],axis=2)
rt_ind = dat[:,1]%40.
fluctuations = np.transpose(np.array([t.T-dat[:,0],d.T-50]),(2,0,1))

cumt = ke.center_around(np.cumsum(t,axis=1),rt_ind)
cumd = ke.center_around(np.cumsum(d,axis=1),rt_ind)
cumdiff = cumt-cumd

rdk,rck,rdk_std,rck_std = ke.kernels(fluctuations,1-dat[:,2],dat[:,3]-1,locked_on_onset=False,RT_ind=rt_ind)
centered_ind = np.arange(cumt.shape[1],dtype=float)
rT = (centered_ind-t.shape[1]+1)*40.

figure()
plot(centered_ind,rdk[0],color='b')
plot(centered_ind,rdk[1],color='r')
fill_between(centered_ind,rdk[0]-rdk_std[0],rdk[0]+rdk_std[0],color='b',alpha=0.3,edgecolor=None)
fill_between(centered_ind,rdk[1]-rdk_std[1],rdk[1]+rdk_std[1],color='r',alpha=0.3,edgecolor=None)

figure()
bins = np.linspace(min(np.nanmin(cumt),np.nanmin(cumd)),max(np.nanmax(cumt),np.nanmax(cumd)),501)
bins2 = np.linspace(np.nanmin(cumdiff),np.nanmax(cumdiff),501)
for i in range(10,16):
	label = "T = %1.0f ms"%(rT[i])
	ht,_ = histogram(cumt[~np.isnan(cumt[:,i]),i],bins)
	hd,_ = histogram(cumd[~np.isnan(cumd[:,i]),i],bins)
	hdiff,_ = histogram(cumdiff[~np.isnan(cumdiff[:,i]),i],bins2)
	subplot(221)
	plot(bins[:-1],ht,label=label)
	subplot(223)
	plot(bins[:-1],hd,label=label)
	subplot(122)
	plot(bins2[:-1],hdiff,label=label)
	legend()

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

figure()
mat_loglike = np.ones((10,t.shape[1]*2+1))
for i,true_sigma in enumerate(np.linspace(5,30,10)):
	model = pe.KnownVarPerfectInference(model_var_t=true_sigma**2,model_var_d=true_sigma**2,prior_mu_t=50.,prior_mu_d=50.,prior_va_t=15**2,prior_va_d=15**2,threshold=1.6,ISI=40)
	mu_t,mu_d,va_t,va_d = model.passiveInference(t,d)
	mu_diff = mu_t-mu_d
	va_diff = va_t+va_d
	prob_miss = normcdf(0.,mu_diff,va_diff)
	obs_likelihood = prob_miss
	obs_likelihood[dat[:,2]==1] = 1.-prob_miss[dat[:,2]==1]
	mat_loglike[i] = np.nansum(np.log(ke.center_around(obs_likelihood,rt_ind+1)),axis=0) # Plus 1 because mu_t, mu_d, va_t and va_d's first element is the prior hyperparameter (i.e. no observations)
imshow(mat_loglike,aspect='auto',interpolation='none',extent=[-1,1,30,5])
colorbar()
