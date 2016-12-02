#ifndef __DecisionPolicy
#define __DecisionPolicy

#include <cmath>
#include <cstddef>
#include <iostream>
#include <cstdio>

#ifdef DEBUG
#ifndef INFO
#define INFO
#endif
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

class DecisionPolicyDescriptor {
protected:
	bool owns_cost;
public:
	double model_var;
	double prior_mu_mean;
	double prior_mu_var;
	int n;
	double dt;
	int nT;
	double dg;
	double T;
	double reward;
	double penalty;
	double iti;
	double tp;
	double* cost;
	int prior_type;
	int n_prior;
	double *mu_prior;
	double *weight_prior;
	
	DecisionPolicyDescriptor(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, bool owns_cost);
	DecisionPolicyDescriptor(double model_var, int n_prior, double* mu_prior, double* weight_prior,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, bool owns_cost);
	
	~DecisionPolicyDescriptor();
};

class DecisionPolicy {
public:
	bool owns_bounds;
	
	int n;
	int nT;
	int bound_strides;
	
	double prior_mu_mean;
	double prior_mu_var;
	double model_var;
	double dt;
	double dg;
	double T;
	double* cost;
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
				   double iti, double tp, double* cost);
	DecisionPolicy(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, double* ub, double* lb,int bound_strides);
	virtual ~DecisionPolicy();
	
	static DecisionPolicy* create(const DecisionPolicyDescriptor& dpc);
	static DecisionPolicy* create(const DecisionPolicyDescriptor& dpc, double* ub, double* lb, int bound_strides);
	
	void disp();
	
	virtual inline double x2g(double t, double x){return NAN;};
	virtual inline double g2x(double t, double g){return NAN;};
	
	virtual double backpropagate_value(){return NAN;};
	virtual double backpropagate_value(double rho, bool compute_bounds){return NAN;};
	virtual double backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2){return NAN;};
	double value_for_root_finding(double rho);
	double iterate_rho_value(double tolerance);
	double iterate_rho_value(double tolerance, double lower_bound, double upper_bound);
	
	double* x_ubound();
	void x_ubound(double* xb);
	double* x_lbound();
	void x_lbound(double* xb);
	double Psi(double mu, double* bound, int itp, double tp, double x0, double t0);
	void rt(double mu, double* g1, double* g2, double* xub, double* xlb);
protected:
	double _model_var;
};

class DecisionPolicyConjPrior : public DecisionPolicy {
public:
	
	DecisionPolicyConjPrior(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost):
			DecisionPolicy(model_var, prior_mu_mean, prior_mu_var, n, dt, T, reward, penalty, iti, tp, cost){};
	DecisionPolicyConjPrior(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, double* ub, double* lb,int bound_strides):
			DecisionPolicy(model_var, prior_mu_mean, prior_mu_var, n, dt, T, reward, penalty, iti, tp, cost, ub, lb, bound_strides){};
	DecisionPolicyConjPrior(const DecisionPolicyDescriptor& dpc);
	DecisionPolicyConjPrior(const DecisionPolicyDescriptor& dpc, double* ub, double* lb, int bound_strides);
	~DecisionPolicyConjPrior();
	
	void disp();
	
	inline double post_mu_var(double t){
		return 1./(t/this->_model_var + 1./this->prior_mu_var);
	}
	
	inline double post_mu_mean(double t, double x){
		return (x/this->_model_var+this->prior_mu_mean/this->prior_mu_var)*this->post_mu_var(t);
	}
	
	inline double x2g(double t, double x){
		return normcdf(this->post_mu_mean(t,x)/sqrt(this->post_mu_var(t)),0.,1.);
	}
	
	inline double g2x(double t, double g){
		return this->_model_var*(normcdfinv(g,0.,1.)/sqrt(this->post_mu_var(t))-this->prior_mu_mean/this->prior_mu_var);
	}
	
	double backpropagate_value();
	double backpropagate_value(double rho, bool compute_bounds);
	double backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2);
};


class DecisionPolicyDiscretePrior : public DecisionPolicy {
protected:
	int n_prior;
	bool is_prior_set;
	double* mu_prior;
	double* mu2_prior;
	double* weight_prior;
	double epsilon;
	double g2x_lower_bound;
	double g2x_upper_bound;
	double g2x_tolerance;
public:
	DecisionPolicyDiscretePrior(double model_var, int n_prior,double* mu_prior, double* weight_prior,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost):
		DecisionPolicy(model_var, 0., 0., n, dt, T, reward, penalty, iti, tp, cost)
	{
		/***
		 * Constructor that shares its bound arrays
		***/
		is_prior_set = false;
		this->epsilon = 1e-10;
		this->set_prior(n_prior,mu_prior,weight_prior);
		this->g2x_tolerance = 1e-12;
		#ifdef DEBUG
		std::cout<<"Created DecisionPolicyDiscretePrior instance at "<<this<<std::endl;
		#endif
	};
	DecisionPolicyDiscretePrior(double model_var, int n_prior,double* mu_prior, double* weight_prior,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, double* ub, double* lb,int bound_strides):
		DecisionPolicy(model_var, 0., 0., n, dt, T, reward, penalty, iti, tp, cost, ub, lb, bound_strides)
	{
		/***
		 * Constructor that shares its bound arrays
		***/
		is_prior_set = false;
		this->epsilon = 1e-10;
		this->set_prior(n_prior,mu_prior,weight_prior);
		this->g2x_tolerance = 1e-12;
		#ifdef DEBUG
		std::cout<<"Created DecisionPolicyDiscretePrior instance at "<<this<<std::endl;
		#endif
	};
	DecisionPolicyDiscretePrior(const DecisionPolicyDescriptor& dpc);
	DecisionPolicyDiscretePrior(const DecisionPolicyDescriptor& dpc, double* ub, double* lb, int bound_strides);
	~DecisionPolicyDiscretePrior();
	
	void set_prior(int n_prior,double* mu_prior, double* weight_prior);
	double get_epsilon(){return this->epsilon;};
	
	void disp();
	
	inline double x2g(double t, double x);
	
	inline void set_g2x_bounds(double lower, double upper){this->g2x_lower_bound = lower; this->g2x_upper_bound = upper;};
	inline void set_g2x_tolerance(double tolerance){this->g2x_tolerance = tolerance;};
	inline double g2x(double t, double g);
	
	double backpropagate_value();
	double backpropagate_value(double rho, bool compute_bounds);
	double backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2);
};
#endif
