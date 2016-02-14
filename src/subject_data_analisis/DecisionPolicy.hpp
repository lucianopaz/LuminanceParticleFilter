#ifndef __DecisionPolicy
#define __DecisionPolicy

double erfinv(double y);

double normcdf(double x, double mu, double v);

double 

def erfinv(y):
a = np.array([ 0.886226899, -1.645349621,  0.914624893, -0.140543331])
b = np.array([-2.118377725,  1.442710462, -0.329097515,  0.012229801])
c = np.array([-1.970840454, -1.624906493,  3.429567803,  1.641345311])
d = np.array([ 3.543889200,  1.637067800])
y0 = 0.7
	if y<-1. or y>1.:
		raise ValueError("erfinv(y) argument out of range [-1.,1]")
	if abs(y)==1.:
		# Precision limit of erf function
		x = y*5.9215871957945083
	elif y<-y0:
		z = math.sqrt(-math.log(0.5*(1.0+y)))
		x = -(((c[3]*z+c[2])*z+c[1])*z+c[0])/((d[1]*z+d[0])*z+1.0)
	else:
		if y<y0:
			z = y*y;
			x = y*(((a[3]*z+a[2])*z+a[1])*z+a[0])/((((b[3]*z+b[3])*z+b[1])*z+b[0])*z+1.0)
		else:
			z = np.sqrt(-math.log(0.5*(1.0-y)))
			x = (((c[3]*z+c[2])*z+c[1])*z+c[0])/((d[1]*z+d[0])*z+1.0)
		# Polish to full accuracy
		x-= (math.erf(x) - y) / (2.0/math.sqrt(math.pi) * math.exp(-x*x));
		x-= (math.erf(x) - y) / (2.0/math.sqrt(math.pi) * math.exp(-x*x));
	return x

def normcdf(x,mu=0.,sigma=1.):
	"""
	Compute normal cummulative distribution with mean mu and standard
	deviation sigma. x, mu and sigma can be a numpy arrays that broadcast
	together.
	
	Syntax:
	y = normcdf(x,mu=0.,sigma=1.)
	"""
	try:
		new_x = (x-mu)/sigma
	except ZeroDivisionError:
		new_x = np.sign(x-mu)*np.inf
	return 0.5 + 0.5*_vectErf(new_x / np.sqrt(2.0))

def normcdfinv(y,mu=0.,sigma=1.):
	"""
	Compute the inverse of the normal cummulative distribution with mean
	mu and standard deviation sigma. y, mu and sigma can be a numpy
	arrays that broadcast together.
	
	Syntax:
	x = normcdfinv(y,mu=0.,sigma=1.)
	"""
	x = np.sqrt(2.0)*_vectErfinv(2.*(y-0.5))
	try:
		iterator = iter(sigma)
	except TypeError:
		if sigma==0.:
			raise ValueError("Invalid sigma supplied to normcdfinv. sigma cannot be 0")
		x = sigma*x+mu
	else:
		if any(sigma==0):
			raise ValueError("Invalid sigma supplied to normcdfinv. sigma cannot be 0")
		x = sigma*x+mu
	return x

class DecisionPolicy {
public:
	int n;
	int nT;
	
	double model_var;
	double prior_mu_mean;
	double prior_mu_var;
	double dt;
	double T;
	double cost;
	double reward;
	double penalty;
	double iti;
	double tp;
	double rho;
	
	double *g;
	double *t;
	double *ub;
	double *lb;
	
	DecisionPolicy();
	~DecisionPolicy();
	
	inline double post_mu_var(double t){
		return 1./(t/this->model_var+1./this->prior_mu_var);
	}
	
	inline double post_mu_mean(double t, double x){
		return (x/this->model_var+this->prior_mu_mean/this->prior_mu_var)*this->post_mu_var(t);
	}
	
	inline double x2g(double t, double x){
		return normcdf(this->post_mu_mean(t,x)/math::sqrt(this->post_mu_var(t)));
	}
	
	inline double g2x(double t, double g){
		return this->model_var*(normcdfinv(g)/math::sqrt(this->post_mu_var(t))-this->prior_mu_mean/this->prior_mu_var);
	}
	
}

#endif
