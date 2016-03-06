#ifndef __DecisionPolicy
#define __DecisionPolicy

#include <cmath>
#include <cstddef>
#include <iostream>

#ifdef DEBUG
#ifndef INFO
#define INFO
#endif
#include <cstdio>
#endif

#define SIGN(x) ((x > 0) - (x < 0))

inline double normcdf(double x, double mu, double sigma){
	if (sigma==0.){
		return x>mu ? INFINITY : -INFINITY;
	}
	return 0.5 + 0.5*erf((x-mu)/sigma*0.70710678118654746);
}

inline double erfinv(double y) {
	double x,z;
	if (y<-1. || y>1.){
		// raise ValueError("erfinv(y) argument out of range [-1.,1]")
		return NAN;
	}
	if (y==1. || y==-1.){
		// Precision limit of erf function
		x = y*5.9215871957945083;
	} else if (y<-0.7){
		z = sqrt(-log(0.5*(1.0+y)));
		x = -(((1.641345311*z+3.429567803)*z-1.624906493)*z-1.970840454)/((1.637067800*z+3.543889200)*z+1.0);
	} else {
		if (y<0.7){
			z = y*y;
			x = y*(((-0.140543331*z+0.914624893)*z-1.645349621)*z+0.886226899)/((((0.012229801*z-0.329097515)*z+1.442710462)*z-2.118377725)*z+1.0);
		} else {
			z = sqrt(-log(0.5*(1.0-y)));
			x = (((1.641345311*z+3.429567803)*z-1.624906493)*z-1.970840454)/((1.637067800*z+3.543889200)*z+1.0);
		}
		// Polish to full accuracy
		x-= (erf(x) - y) / (1.128379167 * exp(-x*x));
		x-= (erf(x) - y) / (1.128379167 * exp(-x*x));
	}
	return x;
}

inline double normcdfinv(double y, double mu, double sigma){
	if (sigma==0.){
		return NAN;
	}
	return 1.4142135623730951*sigma*erfinv(2.*(y-0.5))+mu;
}

class DecisionPolicy {
public:
	int n;
	int nT;
	
	double model_var;
	double prior_mu_mean;
	double prior_mu_var;
	double dt;
	double dg;
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
	
	DecisionPolicy(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double cost);
	~DecisionPolicy();
	
	void disp();
	
	inline double post_mu_var(double t){
		return 1./(t/this->model_var + 1./this->prior_mu_var);
	}
	
	inline double post_mu_mean(double t, double x){
		return (x/this->model_var+this->prior_mu_mean/this->prior_mu_var)*this->post_mu_var(t);
	}
	
	inline double x2g(double t, double x){
		return normcdf(this->post_mu_mean(t,x)/sqrt(this->post_mu_var(t)),0.,1.);
	}
	
	inline double g2x(double t, double g){
		return this->model_var*(normcdfinv(g,0.,1.)/sqrt(this->post_mu_var(t))-this->prior_mu_mean/this->prior_mu_var);
	}
	
	double backpropagate_value();
	double backpropagate_value(double rho, bool compute_bounds);
	double value_for_root_finding(double rho);
	double iterate_rho_value(double tolerance);
	
	double* x_ubound();
	void x_ubound(double* xb);
	double* x_lbound();
	void x_lbound(double* xb);
	double Psi(double mu, double* bound, int itp, double tp, double x0, double t0);
	void rt(double mu, double* g1, double* g2, double* xub, double* xlb);
};

#endif
