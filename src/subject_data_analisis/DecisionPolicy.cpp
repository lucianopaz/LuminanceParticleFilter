#include "DecisionPolicy.hpp"

DecisionPolicyDescriptor::DecisionPolicyDescriptor(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double cost){
	this->model_var = model_var;
	this->prior_mu_mean = prior_mu_mean;
	this->prior_mu_var = prior_mu_var;
	this->mu_prior = (double*)0;
	this->weight_prior = (double*)0;
	this->n = n;
	this->dt = dt;
	this->nT = (int)(T/dt)+1;
	this->T = T;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->cost = cost;
	this->prior_type = 1; // Conjugate prior
}

DecisionPolicyDescriptor::DecisionPolicyDescriptor(double model_var, int n_prior, double* mu_prior, double* weight_prior,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double cost){
	this->model_var = model_var;
	this->n_prior = n_prior;
	this->mu_prior = new double[n_prior];
	this->weight_prior = new double[n_prior];
	this->prior_mu_mean = 0.;
	for (int i=0;i<n_prior;++i){
		this->mu_prior[i] = mu_prior[i];
		this->weight_prior[i] = weight_prior[i];
	}
	this->n = n;
	this->dt = dt;
	this->nT = (int)(T/dt)+1;
	this->T = T;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->cost = cost;
	this->prior_type = 2; // Symmetric discrete prior
}

DecisionPolicyDescriptor::~DecisionPolicyDescriptor(){
	if (this->prior_type==2){
		delete[] this->mu_prior;
		delete[] this->weight_prior;
	}
}

DecisionPolicy::DecisionPolicy(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double cost){
	/***
	 * Constructor that creates its own bound arrays
	***/
	int i;
	
	this->model_var = model_var;
	this->_model_var = model_var;
	this->prior_mu_mean = prior_mu_mean;
	this->prior_mu_var = prior_mu_var;
	if (n%2==0){
		this->n = n+1;
	} else {
		this->n = n;
	}
	this->dt = dt;
	this->T = T;
	this->nT = (int)(T/dt)+1;
	this->cost = cost;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->rho = 0.;
	this->dg = 1./double(n);
	this->g = new double[n];
	for (i=0;i<n;++i){
		this->g[i] = (0.5+double(i))*this->dg;
	}
	this->t = new double[nT];
	for (i=0;i<nT;++i){
		this->t[i] = double(i)*dt;
	}
	this->owns_bounds = true;
	this->ub = new double[nT];
	this->lb = new double[nT];
	this->bound_strides = 1;
}

DecisionPolicy::DecisionPolicy(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double cost, double* ub, double* lb, int bound_strides){
	/***
	 * Constructor that shares its bound arrays
	***/
	int i;
	
	this->model_var = model_var;
	this->_model_var = model_var;
	this->prior_mu_mean = prior_mu_mean;
	this->prior_mu_var = prior_mu_var;
	if (n%2==0){
		this->n = n+1;
	} else {
		this->n = n;
	}
	this->dt = dt;
	this->T = T;
	this->nT = (int)(T/dt)+1;
	this->cost = cost;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->rho = 0.;
	this->dg = 1./double(n);
	this->g = new double[n];
	for (i=0;i<n;++i){
		this->g[i] = (0.5+double(i))*this->dg;
	}
	this->t = new double[nT];
	for (i=0;i<nT;++i){
		this->t[i] = double(i)*dt;
	}
	this->owns_bounds = false;
	this->ub = ub;
	this->lb = lb;
	this->bound_strides = bound_strides;
}

DecisionPolicy::~DecisionPolicy(){
	/***
	 * Destructor
	***/
	delete[] g;
	delete[] t;
	if (owns_bounds){
		delete[] ub;
		delete[] lb;
	}
	#ifdef DEBUG
	std::cout<<"Destroyed DecisionPolicy instance"<<std::endl;
	#endif
}

DecisionPolicy* DecisionPolicy::create(const DecisionPolicyDescriptor& dpc){
	if (dpc.prior_type==1){
		return new DecisionPolicyConjPrior(dpc.model_var, dpc.prior_mu_mean, dpc.prior_mu_var,
					   dpc.n, dpc.dt, dpc.T, dpc.reward, dpc.penalty,
					   dpc.iti, dpc.tp, dpc.cost);
	} else {
		return new DecisionPolicyDiscretePrior(dpc.model_var, dpc.n_prior, dpc.mu_prior, dpc.weight_prior,
					   dpc.n, dpc.dt, dpc.T, dpc.reward, dpc.penalty,
					   dpc.iti, dpc.tp, dpc.cost);
	}
}

DecisionPolicy* DecisionPolicy::create(const DecisionPolicyDescriptor& dpc, double* ub, double* lb, int bound_strides){
	if (dpc.prior_type==1){
		return new DecisionPolicyConjPrior(dpc.model_var, dpc.prior_mu_mean, dpc.prior_mu_var,
					   dpc.n, dpc.dt, dpc.T, dpc.reward, dpc.penalty,
					   dpc.iti, dpc.tp, dpc.cost, ub, lb, bound_strides);
	} else {
		return new DecisionPolicyDiscretePrior(dpc.model_var, dpc.n_prior, dpc.mu_prior, dpc.weight_prior,
					   dpc.n, dpc.dt, dpc.T, dpc.reward, dpc.penalty,
					   dpc.iti, dpc.tp, dpc.cost, ub, lb, bound_strides);
	}
}

double DecisionPolicy::value_for_root_finding(double rho){
	/***
	 * Function that serves as a proxy for the value root finding that
	 * determines rho. Is the same as backpropagate_value(rho,false)
	***/
	return this->backpropagate_value(rho,false);
}

double DecisionPolicy::iterate_rho_value(double tolerance){
	// Use arbitrary default upper and lower bounds
	return this->iterate_rho_value(tolerance,-10.,10.);
}
//~ 
double DecisionPolicy::iterate_rho_value(double tolerance, double lower_bound, double upper_bound){
	/***
	 * Function that implements Brent's algorithm for root finding.
	 * This function was adapted from brent.cpp written by John Burkardt,
	 * that was based on a fortran 77 implementation by Richard Brent.
	 * It searches for the value of rho that sets the value of g=0.5 at
	 * t=0 equal to 0. It finds a value for rho within a certain tolerance
	 * for the value of g=0.5 at t=0.
	***/
	double low_buffer;
	double func_at_low_bound, func_at_up_bound, func_at_low_buffer;
	double d;
	double interval;
	double m;
	double machine_eps = 2.220446049250313E-016;
	double p;
	double q;
	double r;
	double s;
	double tol;
	func_at_low_bound = this->value_for_root_finding(lower_bound);
	func_at_up_bound = this->value_for_root_finding(upper_bound);
	if (func_at_low_bound==0){
		this->rho = lower_bound;
	} else if (func_at_up_bound==0){
		this->rho = upper_bound;
	} else {
		// Adapt the bounds to get a sign changing interval
		while (SIGN(func_at_low_bound)==SIGN(func_at_up_bound)){
			if ((func_at_low_bound<func_at_up_bound and func_at_low_bound<0) ||
			    (func_at_low_bound>func_at_up_bound and func_at_low_bound>0)){
				lower_bound = upper_bound;
				upper_bound*=10;
				if (upper_bound>0){
					upper_bound*=10;
				} else if (upper_bound<0) {
					upper_bound*=-1;
				} else {
					upper_bound=1e-6;
				}
				func_at_up_bound = this->value_for_root_finding(upper_bound);
			} else if ((func_at_low_bound>func_at_up_bound and func_at_low_bound<0) ||
			    (func_at_low_bound<func_at_up_bound and func_at_low_bound>0)){
				upper_bound = lower_bound;
				if (lower_bound<0){
					lower_bound*=10;
				} else if (lower_bound>0) {
					lower_bound*=-1;
				} else {
					lower_bound=-1e-6;
				}
				func_at_low_bound = this->value_for_root_finding(lower_bound);
			}
		}
		// Brent's Algorithm for root finding
		low_buffer = lower_bound;
		func_at_low_buffer = func_at_low_bound;
		interval = upper_bound - lower_bound;
		d = interval;
		
		for ( ; ; ) {
			if (std::abs(func_at_low_buffer)<std::abs(func_at_up_bound)) {
				lower_bound = upper_bound;
				upper_bound = low_buffer;
				low_buffer = lower_bound;
				func_at_low_bound = func_at_up_bound;
				func_at_up_bound = func_at_low_buffer;
				func_at_low_buffer = func_at_low_bound;
			}
			tol = 2.0*machine_eps*std::abs(upper_bound) + tolerance;
			m = 0.5*(low_buffer-upper_bound);
			if (std::abs(m)<= tol || func_at_up_bound==0.0) {
				break;
			}
			if (std::abs(interval)<tol || std::abs(func_at_low_bound)<=std::abs(func_at_up_bound)) {
				interval = m;
				d = interval;
			} else {
				s = func_at_up_bound / func_at_low_bound;
				if (lower_bound==low_buffer){
					p = 2.0 * m * s;
					q = 1.0 - s;
				} else {
					q = func_at_low_bound / func_at_low_buffer;
					r = func_at_up_bound / func_at_low_buffer;
					p = s * ( 2.0 * m * q * ( q - r ) - ( upper_bound - lower_bound ) * ( r - 1.0 ) );
					q = ( q - 1.0 ) * ( r - 1.0 ) * ( s - 1.0 );
				}
				if ( 0.0 < p ) {
					q = - q;
				} else {
					p = - p;
				}
				
				s = interval;
				interval = d;
				
				if ( 2.0 * p < 3.0 * m * q - std::abs ( tol * q ) &&
					p < std::abs ( 0.5 * s * q ) ) {
					d = p / q;
				} else {
					interval = m;
					d = interval;
				}
			}
			lower_bound = upper_bound;
			func_at_low_bound = func_at_up_bound;

			if ( tol < std::abs ( d ) ){
				upper_bound = upper_bound + d;
			} else if ( 0.0 < m ) {
				upper_bound = upper_bound + tol;
			} else {
				upper_bound = upper_bound - tol;
			}
			
			func_at_up_bound = value_for_root_finding(upper_bound);
			if ( (0.0<func_at_up_bound && 0.0<func_at_low_buffer) || (func_at_up_bound<=0.0 && func_at_low_buffer<=0.0)) {
				low_buffer = lower_bound;
				func_at_low_buffer = func_at_low_bound;
				interval = upper_bound - lower_bound;
				d = interval;
			}
		}
		this->rho = upper_bound;
	}
	return this->rho;
}

double* DecisionPolicy::x_ubound(){
	/***
	 * Compute the x space upper bound as a function of time. This
	 * function creates a new double[] and returns it.
	***/
	#ifdef DEBUG
	std::cout<<"Entered x_ubound = "<<std::endl;
	#endif
	int i;
	double *xb = new double[nT];
	for (i=0;i<nT;++i){
		xb[i] = g2x(t[i],ub[i]);
	}
	return xb;
}
void DecisionPolicy::x_ubound(double* xb){
	/***
	 * Compute the x space upper bound as a function of time. This
	 * function places the values in the provided pointer. Beware of the
	 * size of the allocated memory as no size checks are performed
	***/
	#ifdef DEBUG
	std::cout<<"Entered x_ubound_double* with xb = "<<xb<<std::endl;
	#endif
	int i;
	for (i=0;i<nT;++i){
		xb[i] = g2x(t[i],ub[i]);
	}
}

double* DecisionPolicy::x_lbound(){
	/***
	 * Compute the x space lower bound as a function of time. This
	 * function creates a new double[] and returns it.
	***/
	#ifdef DEBUG
	std::cout<<"Entered x_lbound"<<std::endl;
	#endif
	int i;
	double *xb = new double[nT];
	for (i=0;i<nT;++i){
		xb[i] = g2x(t[i],lb[i]);
	}
	return xb;
}
void DecisionPolicy::x_lbound(double* xb){
	/***
	 * Compute the x space lower bound as a function of time. This
	 * function places the values in the provided pointer. Beware of the
	 * size of the allocated memory as no size checks are performed
	***/
	#ifdef DEBUG
	std::cout<<"Entered x_lbound_double* with xb = "<<xb<<std::endl;
	#endif
	int i;
	for (i=0;i<nT;++i){
		xb[i] = g2x(t[i],lb[i]);
	}
}

double DecisionPolicy::Psi(double mu, double* bound, int itp, double tp, double x0, double t0){
	double normpdf = 0.3989422804014327*exp(-0.5*pow(bound[itp]-x0-mu*(tp-t0),2)/this->_model_var/(tp-t0))/sqrt(this->_model_var*(tp-t0));
	double bound_prime;
	if (itp<this->nT-1){
		bound_prime = 0.5*(bound[itp+1]-bound[itp-1])/this->dt;
	} else {
		bound_prime = 0.;
	}
	// double bound_prime = itp<int(sizeof(bound)/sizeof(double)-1) ? (bound[itp+1]-bound[itp])/this->dt : 0.;
	return 0.5*normpdf*(bound_prime-(bound[itp]-x0)/(tp-t0));
}

void DecisionPolicy::rt(double mu, double* g1, double* g2, double* xub, double* xlb){
	#ifdef DEBUG
	std::cout<<"Entered rt"<<std::endl;
	#endif
	unsigned int i,j;
	unsigned int tnT = this->nT;
	double t0,tj,ti,normalization;
	bool delete_xub = false;
	bool delete_xlb = false;
	bool bounds_touched = false;
	if (xub==NULL){
		xub = this->x_ubound();
		delete_xub = true;
	}
	if (xlb==NULL){
		xlb = this->x_lbound();
		delete_xlb = true;
	}
	
	g1[0] = 0.;
	g2[0] = 0.;
	t0 = this->t[0];
	
	if (xub[1]<=xlb[1]){
		// If the bounds collapse to 0 instantly, the decision will be taken instantly and randomly
		bounds_touched = true;
		g1[1] = 0.5;
		g2[1] = 0.5;
	} else {
		g1[1] = -2.*this->Psi(mu,xub,1,this->t[1],this->prior_mu_mean,t0);
		g2[1] = 2.*this->Psi(mu,xlb,1,this->t[1],this->prior_mu_mean,t0);
	}
	// Because of numerical instabilities, we must take care that g1 and g2 are always positive
	if (g1[1]<0.) g1[1] = 0.;
	if (g2[1]<0.) g2[1] = 0.;
	normalization = g1[1]+g2[1];
	for (i=2;i<tnT;++i){
		if (bounds_touched){
			g1[i] = 0.;
			g2[i] = 0.;
		} else {
			ti = this->t[i];
			g1[i] = -this->Psi(mu,xub,i,ti,this->prior_mu_mean,t0);
			g2[i] = this->Psi(mu,xlb,i,ti,this->prior_mu_mean,t0);
			for (j=1;j<i;++j){
				tj = this->t[j];
				g1[i]+=this->dt*(g1[j]*this->Psi(mu,xub,i,ti,xub[j],tj)+
								 g2[j]*this->Psi(mu,xub,i,ti,xlb[j],tj));
				g2[i]-=this->dt*(g1[j]*this->Psi(mu,xlb,i,ti,xub[j],tj)+
								 g2[j]*this->Psi(mu,xlb,i,ti,xlb[j],tj));
			}
			g1[i]*=2.;
			g2[i]*=2.;
			// Because of numerical instabilities, we must take care that g1 and g2 are always positive
			if (g1[i]<0.) g1[i] = 0.;
			if (g2[i]<0.) g2[i] = 0.;
			normalization+= g1[i]+g2[i];
		}
		if (xub[i]<=xlb[i]){
			bounds_touched = true;
		}
	}
	normalization*=this->dt;
	for (i=0;i<tnT;++i){
		g1[i]/=normalization;
		g2[i]/=normalization;
	}
	
	if (delete_xub) delete[] xub;
	if (delete_xlb) delete[] xlb;
}

/***
 * DecisionPolicyConjPrior is a class that implements the dynamic programing method
 * that computes the value of a given belief state as a function of time.
 * 
 * This class implements the method given in Drugowitsch et al 2012 but
 * is limited to constant cost values.
***/

DecisionPolicyConjPrior::~DecisionPolicyConjPrior(){
	/***
	 * Destructor
	***/
	#ifdef DEBUG
	std::cout<<"Destroyed DecisionPolicyConjPrior instance"<<std::endl;
	#endif
}

void DecisionPolicyConjPrior::disp(){
	/***
	 * Print DecisionPolicyConjPrior instance's information
	***/
	std::cout<<"model_var = "<<model_var<<std::endl;
	std::cout<<"_model_var = "<<_model_var<<std::endl;
	std::cout<<"prior_mu_mean = "<<prior_mu_mean<<std::endl;
	std::cout<<"prior_mu_var = "<<prior_mu_var<<std::endl;
	std::cout<<"dt = "<<dt<<std::endl;
	std::cout<<"dg = "<<dg<<std::endl;
	std::cout<<"T = "<<T<<std::endl;
	std::cout<<"cost = "<<cost<<std::endl;
	std::cout<<"reward = "<<reward<<std::endl;
	std::cout<<"iti = "<<iti<<std::endl;
	std::cout<<"tp = "<<tp<<std::endl;
	std::cout<<"rho = "<<rho<<std::endl;
	std::cout<<"n = "<<n<<std::endl;
	std::cout<<"nT = "<<nT<<std::endl;
	std::cout<<"t = "<<t<<std::endl;
	
	std::cout<<"owns_bounds = "<<owns_bounds<<std::endl;
	std::cout<<"bound_strides = "<<bound_strides<<std::endl;
	std::cout<<"ub = "<<ub<<std::endl;
	std::cout<<"lb = "<<lb<<std::endl;
}

double DecisionPolicyConjPrior::backpropagate_value(){
	/***
	 * Mute backpropagate_value function is equivalent to backpropagate_value(this->rho,true)
	***/
	return this->backpropagate_value(this->rho,true);
}

double DecisionPolicyConjPrior::backpropagate_value(double rho, bool compute_bounds){
	/***
	 * Main function:
	 * backpropagate_value(double rho, bool compute_bounds)
	 * 
	 * This function applies dynamic programing to determine the value
	 * of holding belief g at time t. It should be used under two different
	 * circumstances.
	 * 1) Iterate rho value until the value of g=0.5 at t=0 is 0
	 * 2) To compute the decision bounds in g space, once rho has been computed
	 * 
	 * This means that the value of belief g at time t is not stored
	 * during the execution. This is done to improve memory usage and
	 * execution time.
	 * 
	 * This function returns the value of g=0.5 at t=0.
	 * If compute_bounds=true it also sets the values of the bound arrays
	 * ub and lb.
	***/
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone;
	int current_value_zone;
	int i, j, k, bound_ind, curr_invg, fut_invg;
	double post_var_t1, post_var_t, norm_p, maxp;
	double value[n], v1[n], v2[n], v_explore[n], p[n];
	double invg[2][n];
	
	this->rho = rho;
	curr_invg = 0;
	fut_invg = 1;
	#ifdef DEBUG
	FILE *details_file = fopen("details.txt","w");
	FILE *prob_file = fopen("prob.txt","w");
	FILE *value_file = fopen("value.txt","w");
	FILE *v_explore_file = fopen("v_explore.txt","w");
	#endif
	// Compute the value at the time limit T, where the subject must decide
	for (i=0;i<n;++i){
		// Value of deciding option 1
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		// Value of deciding option 2
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		// Value of the belief g[i]
		value[i] = v1[i]>=v2[i] ? v1[i] : v2[i];
		// We compute invg that is the x(t) that corresponds to having g[i] at time T
		// and store it to save computations
		invg[fut_invg][i] = g2x(t[nT-1],g[i]);
		#ifdef DEBUG
		if (i<n-1){
			fprintf(value_file,"%f\t",value[i]);
		} else {
			fprintf(value_file,"%f\n",value[i]);
		}
		#endif
		
		if (compute_bounds){
			if (v1[i]==v2[i]){
				// If the values are the same at a given g, then that g is the decision bound
				this->lb[bound_strides*(nT-1)] = g[i];
				this->ub[bound_strides*(nT-1)] = g[i];
			} else if (i>0 && i<n){
				// If the values do not match, we interpolate the g where the values cross
				if ((v1[i]>v2[i] && v1[i-1]<v2[i-1]) || (v1[i]<v2[i] && v1[i-1]>v2[i-1])){
					lb[bound_strides*(nT-1)] = ((v1[i-1]-v2[i-1])*g[i] + (v1[i]-v2[i])*g[i-1]) / (v2[i]-v1[i]+v1[i-1]-v2[i-1]);
					ub[bound_strides*(nT-1)] = lb[nT-1];
				}
			}
		}
	}
	
	post_var_t1 = post_mu_var(this->t[nT-1]);
	// Dynamic programing loop that goes backward in time from T->0
	// Speed increase by precalculating values
	const double prior_div = prior_mu_mean/prior_mu_var;
	const double inv_model_var = 1./_model_var;
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		bound_ind = bound_strides*i;
		ub[bound_ind] = 1.;
		lb[bound_ind] = 0.;
		
		//Speed increase by reducing array access
		const double t_i = t[i];
		post_var_t = post_mu_var(t_i);
		for (j=0;j<n;++j){
			v_explore[j] = 0.;
			invg[curr_invg][j] = g2x(t_i,g[j]);
			norm_p = 0.;
			maxp = -INFINITY;
			// Speed increase by reducing array access and precalculating values
			const double mu_n_dt = post_mu_mean(t_i,invg[curr_invg][j])*dt;
			const double mean_1 = invg[curr_invg][j]+mu_n_dt;
			const double var_1 = 1./((post_var_t*dt+_model_var)*dt);
			const double* future_x = invg[fut_invg];
			
			// Compute P(g(t+dt)|g(t)) in two steps. First compute the exponent
			for (k=0;k<n;++k){
				p[k] = -0.5*pow(future_x[k]-mean_1,2)*var_1+
						0.5*pow(future_x[k]*inv_model_var+prior_div,2)*post_var_t1;
				maxp = p[k]>maxp ? p[k] : maxp;
				#ifdef DEBUG
				fprintf(details_file,"%f\t%f\t%f\t%f\t%f\n",invg[fut_invg][k],invg[curr_invg][j],mu_n_dt,post_var_t,post_var_t1);
				#endif
			}
			// Then exponentiate and compute the value of exploring
			for (k=0;k<n;++k){
				p[k] = exp(p[k]-maxp);
				norm_p+=p[k];
				v_explore[j]+= p[k]*value[k];
			}
			// Divide the value of exploring by the normalization factor and discount the cost and rho
			v_explore[j] = v_explore[j]/norm_p - (cost+rho)*dt;
			
			#ifdef DEBUG
			for (k=0;k<n-1;++k){
				fprintf(prob_file,"%f\t",p[k]/norm_p);
			}
			fprintf(prob_file,"%f\n",p[k]/norm_p);
			if (j<n-1){
				fprintf(v_explore_file,"%f\t",v_explore[j]);
			} else {
				fprintf(v_explore_file,"%f\n",v_explore[j]);
			}
			#endif
		}
		// Update temporal values
		post_var_t1 = post_var_t;
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		// Value computation
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;++j){
			if (v1[j]>=v2[j] && v1[j]>=v_explore[j]){
				value[j] = v1[j];
				current_value_zone = 1;
			} else if (v2[j]>v1[j] && v2[j]>=v_explore[j]){
				value[j] = v2[j];
				current_value_zone = 2;
			} else if (v_explore[j]>v1[j] && v_explore[j]>v2[j]){
				value[j] = v_explore[j];
				current_value_zone = 0;
			}
			#ifdef DEBUG
			if (j<n-1){
				fprintf(value_file,"%f\t",value[j]);
			} else {
				fprintf(value_file,"%f\n",value[j]);
			}
			#endif
			// Bound computation
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j])<1e-8){
						if (!setted_ub){
							ub[bound_ind] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j])<1e-8){
						lb[bound_ind] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[bound_ind] = (g[j-1]*(v1[j]-v_explore[j]) - g[j]*(v1[j-1]-v_explore[j-1])) / (v_explore[j-1]-v_explore[j]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[bound_ind] = lb[bound_ind];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v_explore[j]-v2[j]) - g[j]*(v_explore[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j]-v_explore[j-1]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	std::cout<<"Exited backpropagate_value "<<std::endl;
	#endif
	return value[int(0.5*n)];
}

double DecisionPolicyConjPrior::backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2){
	/***
	 * Main function:
	 * backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2)
	 * 
	 * This function applies dynamic programing to determine the value
	 * of holding belief g at time t. It should be used under two different
	 * circumstances.
	 * 1) Iterate rho value until the value of g=0.5 at t=0 is 0
	 * 2) To compute the decision bounds in g space, once rho has been computed
	 * 
	 * This means that the value of belief g at time t is not stored
	 * during the execution. This is done to improve memory usage and
	 * execution time.
	 * 
	 * This function returns the value of g=0.5 at t=0.
	 * If compute_bounds=true it also sets the values of the bound arrays
	 * ub and lb.
	***/
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone;
	int current_value_zone;
	int i, j, k, bound_ind, curr_invg, fut_invg;
	double post_var_t1, post_var_t, norm_p, maxp;
	double p[n];
	double invg[2][n];
	
	this->rho = rho;
	curr_invg = 0;
	fut_invg = 1;
	#ifdef DEBUG
	FILE *details_file = fopen("details.txt","w");
	FILE *prob_file = fopen("prob.txt","w");
	FILE *value_file = fopen("value.txt","w");
	FILE *v_explore_file = fopen("v_explore.txt","w");
	#endif
	// Compute the value at the time limit T, where the subject must decide
	for (i=0;i<n;++i){
		// Value of deciding option 1
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		// Value of deciding option 2
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		// Value of the belief g[i]
		value[i+(nT-1)*n] = v1[i]>=v2[i] ? v1[i] : v2[i];
		// We compute invg that is the x(t) that corresponds to having g[i] at time T
		// and store it to save computations
		invg[fut_invg][i] = g2x(t[nT-1],g[i]);
		#ifdef DEBUG
		if (i<n-1){
			fprintf(value_file,"%f\t",value[i+(nT-1)*n]);
		} else {
			fprintf(value_file,"%f\n",value[i+(nT-1)*n]);
		}
		#endif
		
		if (compute_bounds){
			if (v1[i]==v2[i]){
				// If the values are the same at a given g, then that g is the decision bound
				this->lb[bound_strides*(nT-1)] = g[i];
				this->ub[bound_strides*(nT-1)] = g[i];
			} else if (i>0 && i<n){
				// If the values do not match, we interpolate the g where the values cross
				if ((v1[i]>v2[i] && v1[i-1]<v2[i-1]) || (v1[i]<v2[i] && v1[i-1]>v2[i-1])){
					lb[bound_strides*(nT-1)] = ((v1[i-1]-v2[i-1])*g[i] + (v1[i]-v2[i])*g[i-1]) / (v2[i]-v1[i]+v1[i-1]-v2[i-1]);
					ub[bound_strides*(nT-1)] = lb[nT-1];
				}
			}
		}
	}
	
	post_var_t1 = post_mu_var(this->t[nT-1]);
	// Dynamic programing loop that goes backward in time from T->0
	// Speed increase by precalculating values
	const double prior_div = prior_mu_mean/prior_mu_var;
	const double inv_model_var = 1./_model_var;
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		bound_ind = bound_strides*i;
		ub[bound_ind] = 1.;
		lb[bound_ind] = 0.;
		
		//Speed increase by reducing array access
		const double t_i = t[i];
		post_var_t = post_mu_var(t_i);
		for (j=0;j<n;++j){
			v_explore[j+i*n] = 0.;
			invg[curr_invg][j] = g2x(t_i,g[j]);
			norm_p = 0.;
			maxp = -INFINITY;
			// Speed increase by reducing array access and precalculating values
			const double mu_n_dt = post_mu_mean(t_i,invg[curr_invg][j])*dt;
			const double mean_1 = invg[curr_invg][j]+mu_n_dt;
			const double var_1 = 1./((post_var_t*dt+_model_var)*dt);
			const double* future_x = invg[fut_invg];
			
			// Compute P(g(t+dt)|g(t)) in two steps. First compute the exponent
			for (k=0;k<n;++k){
				p[k] = -0.5*pow(future_x[k]-mean_1,2)*var_1+
						0.5*pow(future_x[k]*inv_model_var+prior_div,2)*post_var_t1;
				maxp = p[k]>maxp ? p[k] : maxp;
				#ifdef DEBUG
				fprintf(details_file,"%f\t%f\t%f\t%f\t%f\n",invg[fut_invg][k],invg[curr_invg][j],mu_n_dt,post_var_t,post_var_t1);
				#endif
			}
			// Then exponentiate and compute the value of exploring
			for (k=0;k<n;++k){
				p[k] = exp(p[k]-maxp);
				norm_p+=p[k];
				v_explore[j+i*n]+= p[k]*value[k+(i+1)*n];
			}
			// Divide the value of exploring by the normalization factor and discount the cost and rho
			v_explore[j+i*n] = v_explore[j+i*n]/norm_p - (cost+rho)*dt;
			
			#ifdef DEBUG
			for (k=0;k<n-1;++k){
				fprintf(prob_file,"%f\t",p[k]/norm_p);
			}
			fprintf(prob_file,"%f\n",p[k]/norm_p);
			if (j<n-1){
				fprintf(v_explore_file,"%f\t",v_explore[j+i*n]);
			} else {
				fprintf(v_explore_file,"%f\n",v_explore[j+i*n]);
			}
			#endif
		}
		// Update temporal values
		post_var_t1 = post_var_t;
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		// Value computation
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;++j){
			if (v1[j]>=v2[j] && v1[j]>=v_explore[j+i*n]){
				value[j+i*n] = v1[j];
				current_value_zone = 1;
			} else if (v2[j]>v1[j] && v2[j]>=v_explore[j+i*n]){
				value[j+i*n] = v2[j];
				current_value_zone = 2;
			} else if (v_explore[j+i*n]>v1[j] && v_explore[j+i*n]>v2[j]){
				value[j+i*n] = v_explore[j+i*n];
				current_value_zone = 0;
			}
			#ifdef DEBUG
			if (j<n-1){
				fprintf(value_file,"%f\t",value[j+i*n]);
			} else {
				fprintf(value_file,"%f\n",value[j+i*n]);
			}
			#endif
			// Bound computation
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j+i*n])<1e-8){
						if (!setted_ub){
							ub[bound_ind] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j+i*n])<1e-8){
						lb[bound_ind] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[bound_ind] = (g[j-1]*(v1[j]-v_explore[j+i*n]) - g[j]*(v1[j-1]-v_explore[j-1+i*n])) / (v_explore[j-1+i*n]-v_explore[j+i*n]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[bound_ind] = lb[bound_ind];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v_explore[j+i*n]-v2[j]) - g[j]*(v_explore[j-1+i*n]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j+i*n]-v_explore[j-1+i*n]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	std::cout<<"Exited backpropagate_value "<<std::endl;
	#endif
	return value[int(0.5*n)];
}



DecisionPolicyDiscretePrior::~DecisionPolicyDiscretePrior(){
	/***
	 * Destructor
	***/
	if (is_prior_set){
		delete[] mu_prior;
		delete[] mu2_prior;
		delete[] weight_prior;
	}
	#ifdef DEBUG
	std::cout<<"Destroyed DecisionPolicyDiscretePrior instance"<<std::endl;
	#endif
}

void DecisionPolicyDiscretePrior::set_prior(int n_prior,double* mu_prior, double* weight_prior){
	int i;
	double normalization = 0.;
	this->n_prior = n_prior;
	if (this->is_prior_set){
		delete[] this->mu_prior;
		delete[] this->mu2_prior;
		delete[] this->weight_prior;
	}
	this->mu_prior = new double[n_prior];
	this->mu2_prior = new double[n_prior];
	this->weight_prior = new double[n_prior];
	this->prior_mu_mean = 0.;
	
	for (i=0;i<n_prior;++i){
		if (mu_prior[i]==0.){
			this->mu_prior[i] = this->epsilon;
		} else {
			this->mu_prior[i] = mu_prior[i];
		}
		this->mu2_prior[i] = this->mu_prior[i]*this->mu_prior[i];
		this->weight_prior[i] = weight_prior[i];
		normalization+= this->weight_prior[i];
	}
	this->prior_mu_var = 0.;
	for (i=0;i<n_prior;++i){
		this->weight_prior[i]*=0.5/normalization;
		this->prior_mu_var+= 2*this->weight_prior[i]*this->mu2_prior[i];
	}
	this->is_prior_set = true;
}

void DecisionPolicyDiscretePrior::disp(){
	/***
	 * Print DecisionPolicyDiscretePrior instance's information
	***/
	std::cout<<"model_var = "<<model_var<<std::endl;
	std::cout<<"_model_var = "<<_model_var<<std::endl;
	std::cout<<"mu_prior = "<<mu_prior<<std::endl;
	std::cout<<"mu2_prior = "<<mu2_prior<<std::endl;
	std::cout<<"weight_prior = "<<weight_prior<<std::endl;
	std::cout<<"dt = "<<dt<<std::endl;
	std::cout<<"dg = "<<dg<<std::endl;
	std::cout<<"T = "<<T<<std::endl;
	std::cout<<"cost = "<<cost<<std::endl;
	std::cout<<"reward = "<<reward<<std::endl;
	std::cout<<"iti = "<<iti<<std::endl;
	std::cout<<"tp = "<<tp<<std::endl;
	std::cout<<"rho = "<<rho<<std::endl;
	std::cout<<"n = "<<n<<std::endl;
	std::cout<<"nT = "<<nT<<std::endl;
	std::cout<<"t = "<<t<<std::endl;
	
	std::cout<<"owns_bounds = "<<owns_bounds<<std::endl;
	std::cout<<"bound_strides = "<<bound_strides<<std::endl;
	std::cout<<"ub = "<<ub<<std::endl;
	std::cout<<"lb = "<<lb<<std::endl;
}

inline double DecisionPolicyDiscretePrior::x2g(double t, double x){
	double num = 0.;
	double den = 0.;
	double exponent_num = 0.;
	double exponent_den = 0.;
	double max_exp=INFINITY;
	for (int i=0;i<n_prior;++i){
		exponent_num = -(0.5*mu2_prior[i]*t+mu_prior[i]*x);
		max_exp = max_exp>exponent_num ? max_exp : exponent_num;
		exponent_den = -(0.5*mu2_prior[i]*t-mu_prior[i]*x);
		max_exp = max_exp>exponent_den ? max_exp : exponent_den;
	}
	for (int i=0;i<n_prior;++i){
		exponent_num = -(0.5*mu2_prior[i]*t+mu_prior[i]*x-max_exp)/_model_var;
		exponent_den = -(0.5*mu2_prior[i]*t-mu_prior[i]*x-max_exp)/_model_var;
		num+= weight_prior[i]*exp(exponent_num);
		den+= weight_prior[i]*exp(exponent_num);
	}
	return 1./(1.+num/den);
}

inline double DecisionPolicyDiscretePrior::g2x(double t, double g){
	/***
	 * Function that implements Brent's algorithm for root finding.
	 * This function was adapted from brent.cpp written by John Burkardt,
	 * that was based on a fortran 77 implementation by Richard Brent.
	***/
	double low_buffer;
	double func_at_low_bound, func_at_up_bound, func_at_low_buffer;
	double d;
	double interval;
	double m;
	double machine_eps = 2.220446049250313E-016;
	double p;
	double q;
	double r;
	double s;
	double tol;
	double tolerance = this->g2x_tolerance;
	double lower_bound = this->g2x_lower_bound;
	double upper_bound = this->g2x_upper_bound;
	func_at_low_bound = this->x2g(t, lower_bound);
	func_at_up_bound = this->x2g(t, upper_bound);
	if (func_at_low_bound==0){
		return lower_bound;
	} else if (func_at_up_bound==0){
		return upper_bound;
	} else {
		// Adapt the bounds to get a sign changing interval
		while (SIGN(func_at_low_bound)==SIGN(func_at_up_bound)){
			if ((func_at_low_bound<func_at_up_bound and func_at_low_bound<0) ||
			    (func_at_low_bound>func_at_up_bound and func_at_low_bound>0)){
				lower_bound = upper_bound;
				upper_bound*=10;
				if (upper_bound>0){
					upper_bound*=10;
				} else if (upper_bound<0) {
					upper_bound*=-1;
				} else {
					upper_bound=1e-6;
				}
				func_at_up_bound = this->x2g(t, upper_bound);
				this->set_g2x_bounds(lower_bound,upper_bound);
			} else if ((func_at_low_bound>func_at_up_bound and func_at_low_bound<0) ||
			    (func_at_low_bound<func_at_up_bound and func_at_low_bound>0)){
				upper_bound = lower_bound;
				if (lower_bound<0){
					lower_bound*=10;
				} else if (lower_bound>0) {
					lower_bound*=-1;
				} else {
					lower_bound=-1e-6;
				}
				func_at_low_bound = this->x2g(t, lower_bound);
				this->set_g2x_bounds(lower_bound,upper_bound);
			}
		}
	}
	// Brent's Algorithm for root finding
	low_buffer = lower_bound;
	interval = upper_bound - lower_bound;
	d = interval;
	
	for ( ; ; ) {
		if (std::abs(func_at_low_buffer)<std::abs(func_at_up_bound)) {
			lower_bound = upper_bound;
			upper_bound = low_buffer;
			low_buffer = lower_bound;
			func_at_low_bound = func_at_up_bound;
			func_at_up_bound = func_at_low_buffer;
			func_at_low_buffer = func_at_low_bound;
		}
		tol = 2.0*machine_eps*std::abs(upper_bound) + tolerance;
		m = 0.5*(low_buffer-upper_bound);
		if (std::abs(m)<= tol || func_at_up_bound==0.0) {
			break;
		}
		if (std::abs(interval)<tol || std::abs(func_at_low_bound)<=std::abs(func_at_up_bound)) {
			interval = m;
			d = interval;
		} else {
			s = func_at_up_bound / func_at_low_bound;
			if (lower_bound==low_buffer){
				p = 2.0 * m * s;
				q = 1.0 - s;
			} else {
				q = func_at_low_bound / func_at_low_buffer;
				r = func_at_up_bound / func_at_low_buffer;
				p = s * ( 2.0 * m * q * ( q - r ) - ( upper_bound - lower_bound ) * ( r - 1.0 ) );
				q = ( q - 1.0 ) * ( r - 1.0 ) * ( s - 1.0 );
			}
			if ( 0.0 < p ) {
				q = - q;
			} else {
				p = - p;
			}
			
			s = interval;
			interval = d;
			
			if ( 2.0 * p < 3.0 * m * q - std::abs ( tol * q ) &&
				p < std::abs ( 0.5 * s * q ) ) {
				d = p / q;
			} else {
				interval = m;
				d = interval;
			}
		}
		lower_bound = upper_bound;
		func_at_low_bound = func_at_up_bound;

		if ( tol < std::abs ( d ) ){
			upper_bound = upper_bound + d;
		} else if ( 0.0 < m ) {
			upper_bound = upper_bound + tol;
		} else {
			upper_bound = upper_bound - tol;
		}
		
		func_at_up_bound = x2g(t, upper_bound);
		if ( (0.0<func_at_up_bound && 0.0<func_at_low_buffer) || (func_at_up_bound<=0.0 && func_at_low_buffer<=0.0)) {
			low_buffer = lower_bound;
			func_at_low_buffer = func_at_low_bound;
			interval = upper_bound - lower_bound;
			d = interval;
		}
	}
	this->set_g2x_bounds(-10.,10.);
	return upper_bound;
}

double DecisionPolicyDiscretePrior::backpropagate_value(){
	/***
	 * Mute backpropagate_value function is equivalent to backpropagate_value(this->rho,true)
	***/
	return this->backpropagate_value(this->rho,true);
}

double DecisionPolicyDiscretePrior::backpropagate_value(double rho, bool compute_bounds){
	/***
	 * Main function:
	 * backpropagate_value(double rho, bool compute_bounds)
	 * 
	 * This function applies dynamic programing to determine the value
	 * of holding belief g at time t. It should be used under two different
	 * circumstances.
	 * 1) Iterate rho value until the value of g=0.5 at t=0 is 0
	 * 2) To compute the decision bounds in g space, once rho has been computed
	 * 
	 * This means that the value of belief g at time t is not stored
	 * during the execution. This is done to improve memory usage and
	 * execution time.
	 * 
	 * This function returns the value of g=0.5 at t=0.
	 * If compute_bounds=true it also sets the values of the bound arrays
	 * ub and lb.
	***/
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone;
	int current_value_zone;
	int i, j, k, bound_ind, curr_invg, fut_invg, ind_prior;
	double norm_p;
	double sum_future_a_plus_b, sum_future_mua, sum_future_mub;
	double sum_future_a, sum_future_b, sum_present_a_plus_b;
	double future_a, future_b;
	double value[n], v1[n], v2[n], v_explore[n], p[n];
	double invg[2][n];
	
	this->rho = rho;
	curr_invg = 0;
	fut_invg = 1;
	#ifdef DEBUG
	FILE *details_file = fopen("details.txt","w");
	FILE *prob_file = fopen("prob.txt","w");
	FILE *value_file = fopen("value.txt","w");
	FILE *v_explore_file = fopen("v_explore.txt","w");
	#endif
	// Compute the value at the time limit T, where the subject must decide
	for (i=0;i<n;++i){
		// Value of deciding option 1
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		// Value of deciding option 2
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		// Value of the belief g[i]
		value[i] = v1[i]>=v2[i] ? v1[i] : v2[i];
		// We compute invg that is the x(t) that corresponds to having g[i] at time T
		// and store it to save computations
		invg[fut_invg][i] = g2x(t[nT-1],g[i]);
		#ifdef DEBUG
		if (i<n-1){
			fprintf(value_file,"%f\t",value[i]);
		} else {
			fprintf(value_file,"%f\n",value[i]);
		}
		#endif
		
		if (compute_bounds){
			if (v1[i]==v2[i]){
				// If the values are the same at a given g, then that g is the decision bound
				this->lb[bound_strides*(nT-1)] = g[i];
				this->ub[bound_strides*(nT-1)] = g[i];
			} else if (i>0 && i<n){
				// If the values do not match, we interpolate the g where the values cross
				if ((v1[i]>v2[i] && v1[i-1]<v2[i-1]) || (v1[i]<v2[i] && v1[i-1]>v2[i-1])){
					lb[bound_strides*(nT-1)] = ((v1[i-1]-v2[i-1])*g[i] + (v1[i]-v2[i])*g[i-1]) / (v2[i]-v1[i]+v1[i-1]-v2[i-1]);
					ub[bound_strides*(nT-1)] = lb[nT-1];
				}
			}
		}
	}
	
	// Dynamic programing loop that goes backward in time from T->0
	// Speed increase by precalculating values
	const double norm_exponent_factor = -0.5/_model_var/dt;
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		bound_ind = bound_strides*i;
		ub[bound_ind] = 1.;
		lb[bound_ind] = 0.;
		
		//Speed increase by reducing array access
		const double t_i = t[i];
		const double t_i1 = t[i+1];
		for (j=0;j<n;++j){
			v_explore[j] = 0.;
			invg[curr_invg][j] = g2x(t_i,g[j]);
			this->set_g2x_bounds(invg[curr_invg][j],this->g2x_upper_bound);
			norm_p = 0.;
			// Speed increase by reducing array access and precalculating values
			const double present_x = invg[curr_invg][j];
			sum_future_a_plus_b = 0.;
			sum_future_mua = 0.;
			sum_future_mub = 0.;
			sum_future_a = 0.;
			sum_future_b = 0.;
			sum_present_a_plus_b = 0.;
			
			// Compute P(g(t+dt)|g(t)) in two steps. First compute the exponent
			for (k=0;k<n;++k){
				const double future_x = invg[fut_invg][k];
				for (ind_prior=0;ind_prior<n_prior;++ind_prior){
					future_a = weight_prior[ind_prior]*exp(future_x*mu_prior[ind_prior]-0.5*t_i1*mu2_prior[ind_prior]);
					future_b = weight_prior[ind_prior]*exp(-future_x*mu_prior[ind_prior]-0.5*t_i1*mu2_prior[ind_prior]);
					sum_future_a_plus_b+= future_a+future_b;
					sum_future_mua+= mu_prior[ind_prior]*future_a;
					sum_future_mub+= mu_prior[ind_prior]*future_b;
					sum_future_a+= future_a;
					sum_future_b+= future_a;
					sum_present_a_plus_b+= weight_prior[ind_prior]*(exp(present_x*mu_prior[ind_prior]-0.5*t_i*mu2_prior[ind_prior])+
																	exp(-present_x*mu_prior[ind_prior]-0.5*t_i*mu2_prior[ind_prior]));
				}
				p[k] = exp(norm_exponent_factor*pow(future_x-present_x,2))*
						pow(sum_future_a_plus_b,3)/(sum_future_mua*sum_future_b+sum_future_a*sum_future_mub)/sum_present_a_plus_b;
				#ifdef DEBUG
				fprintf(details_file,"%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",future_x,present_x,sum_future_a_plus_b,sum_future_mua,sum_future_mub,sum_future_a,sum_future_b,sum_present_a_plus_b);
				#endif
			}
			// Then exponentiate and compute the value of exploring
			for (k=0;k<n;++k){
				norm_p+=p[k];
				v_explore[j]+= p[k]*value[k];
			}
			// Divide the value of exploring by the normalization factor and discount the cost and rho
			v_explore[j] = v_explore[j]/norm_p - (cost+rho)*dt;
			
			#ifdef DEBUG
			for (k=0;k<n-1;++k){
				fprintf(prob_file,"%f\t",p[k]/norm_p);
			}
			fprintf(prob_file,"%f\n",p[k]/norm_p);
			if (j<n-1){
				fprintf(v_explore_file,"%f\t",v_explore[j]);
			} else {
				fprintf(v_explore_file,"%f\n",v_explore[j]);
			}
			#endif
		}
		// Update temporal values
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		// Value computation
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;++j){
			if (v1[j]>=v2[j] && v1[j]>=v_explore[j]){
				value[j] = v1[j];
				current_value_zone = 1;
			} else if (v2[j]>v1[j] && v2[j]>=v_explore[j]){
				value[j] = v2[j];
				current_value_zone = 2;
			} else if (v_explore[j]>v1[j] && v_explore[j]>v2[j]){
				value[j] = v_explore[j];
				current_value_zone = 0;
			}
			#ifdef DEBUG
			if (j<n-1){
				fprintf(value_file,"%f\t",value[j]);
			} else {
				fprintf(value_file,"%f\n",value[j]);
			}
			#endif
			// Bound computation
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j])<1e-8){
						if (!setted_ub){
							ub[bound_ind] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j])<1e-8){
						lb[bound_ind] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[bound_ind] = (g[j-1]*(v1[j]-v_explore[j]) - g[j]*(v1[j-1]-v_explore[j-1])) / (v_explore[j-1]-v_explore[j]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[bound_ind] = lb[bound_ind];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v_explore[j]-v2[j]) - g[j]*(v_explore[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j]-v_explore[j-1]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	std::cout<<"Exited backpropagate_value "<<std::endl;
	#endif
	return value[int(0.5*n)];
}

double DecisionPolicyDiscretePrior::backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2){
	/***
	 * Main function:
	 * backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2)
	 * 
	 * This function applies dynamic programing to determine the value
	 * of holding belief g at time t. It should be used under two different
	 * circumstances.
	 * 1) Iterate rho value until the value of g=0.5 at t=0 is 0
	 * 2) To compute the decision bounds in g space, once rho has been computed
	 * 
	 * This means that the value of belief g at time t is not stored
	 * during the execution. This is done to improve memory usage and
	 * execution time.
	 * 
	 * This function returns the value of g=0.5 at t=0.
	 * If compute_bounds=true it also sets the values of the bound arrays
	 * ub and lb.
	***/
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone;
	int current_value_zone;
	int i, j, k, bound_ind, curr_invg, fut_invg, ind_prior;
	double norm_p;
	double sum_future_a_plus_b, sum_future_mua, sum_future_mub;
	double sum_future_a, sum_future_b, sum_present_a_plus_b;
	double future_a, future_b;
	double p[n];
	double invg[2][n];
	
	this->rho = rho;
	curr_invg = 0;
	fut_invg = 1;
	#ifdef DEBUG
	FILE *details_file = fopen("details.txt","w");
	FILE *prob_file = fopen("prob.txt","w");
	FILE *value_file = fopen("value.txt","w");
	FILE *v_explore_file = fopen("v_explore.txt","w");
	#endif
	// Compute the value at the time limit T, where the subject must decide
	for (i=0;i<n;++i){
		// Value of deciding option 1
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		// Value of deciding option 2
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		// Value of the belief g[i]
		value[i+(nT-1)*n] = v1[i]>=v2[i] ? v1[i] : v2[i];
		// We compute invg that is the x(t) that corresponds to having g[i] at time T
		// and store it to save computations
		invg[fut_invg][i] = g2x(t[nT-1],g[i]);
		#ifdef DEBUG
		if (i<n-1){
			fprintf(value_file,"%f\t",value[i+(nT-1)*n]);
		} else {
			fprintf(value_file,"%f\n",value[i+(nT-1)*n]);
		}
		#endif
		
		if (compute_bounds){
			if (v1[i]==v2[i]){
				// If the values are the same at a given g, then that g is the decision bound
				this->lb[bound_strides*(nT-1)] = g[i];
				this->ub[bound_strides*(nT-1)] = g[i];
			} else if (i>0 && i<n){
				// If the values do not match, we interpolate the g where the values cross
				if ((v1[i]>v2[i] && v1[i-1]<v2[i-1]) || (v1[i]<v2[i] && v1[i-1]>v2[i-1])){
					lb[bound_strides*(nT-1)] = ((v1[i-1]-v2[i-1])*g[i] + (v1[i]-v2[i])*g[i-1]) / (v2[i]-v1[i]+v1[i-1]-v2[i-1]);
					ub[bound_strides*(nT-1)] = lb[nT-1];
				}
			}
		}
	}
	
	// Dynamic programing loop that goes backward in time from T->0
	// Speed increase by precalculating values
	const double norm_exponent_factor = -0.5/_model_var/dt;
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		bound_ind = bound_strides*i;
		ub[bound_ind] = 1.;
		lb[bound_ind] = 0.;
		
		//Speed increase by reducing array access
		const double t_i = t[i];
		const double t_i1 = t[i+1];
		for (j=0;j<n;++j){
			v_explore[j+i*n] = 0.;
			invg[curr_invg][j] = g2x(t_i,g[j]);
			norm_p = 0.;
			// Speed increase by reducing array access and precalculating values
			const double present_x = invg[curr_invg][j];
			sum_future_a_plus_b = 0.;
			sum_future_mua = 0.;
			sum_future_mub = 0.;
			sum_future_a = 0.;
			sum_future_b = 0.;
			sum_present_a_plus_b = 0.;
			
			// Compute P(g(t+dt)|g(t)) in two steps. First compute the exponent
			for (k=0;k<n;++k){
				const double future_x = invg[fut_invg][k];
				for (ind_prior=0;ind_prior<n_prior;++ind_prior){
					future_a = weight_prior[ind_prior]*exp(future_x*mu_prior[ind_prior]-0.5*t_i1*mu2_prior[ind_prior]);
					future_b = weight_prior[ind_prior]*exp(-future_x*mu_prior[ind_prior]-0.5*t_i1*mu2_prior[ind_prior]);
					sum_future_a_plus_b+= future_a+future_b;
					sum_future_mua+= mu_prior[ind_prior]*future_a;
					sum_future_mub+= mu_prior[ind_prior]*future_b;
					sum_future_a+= future_a;
					sum_future_b+= future_a;
					sum_present_a_plus_b+= weight_prior[ind_prior]*(exp(present_x*mu_prior[ind_prior]-0.5*t_i*mu2_prior[ind_prior])+
																	exp(-present_x*mu_prior[ind_prior]-0.5*t_i*mu2_prior[ind_prior]));
				}
				p[k] = exp(norm_exponent_factor*pow(future_x-present_x,2))*
						pow(sum_future_a_plus_b,3)/(sum_future_mua*sum_future_b+sum_future_a*sum_future_mub)/sum_present_a_plus_b;
				#ifdef DEBUG
				fprintf(details_file,"%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",future_x,present_x,sum_future_a_plus_b,sum_future_mua,sum_future_mub,sum_future_a,sum_future_b,sum_present_a_plus_b);
				#endif
			}
			// Then exponentiate and compute the value of exploring
			for (k=0;k<n;++k){
				norm_p+=p[k];
				v_explore[j+i*n]+= p[k]*value[k+(i+1)*n];
			}
			// Divide the value of exploring by the normalization factor and discount the cost and rho
			v_explore[j+i*n] = v_explore[j+i*n]/norm_p - (cost+rho)*dt;
			
			#ifdef DEBUG
			for (k=0;k<n-1;++k){
				fprintf(prob_file,"%f\t",p[k]/norm_p);
			}
			fprintf(prob_file,"%f\n",p[k]/norm_p);
			if (j<n-1){
				fprintf(v_explore_file,"%f\t",v_explore[j+i*n]);
			} else {
				fprintf(v_explore_file,"%f\n",v_explore[j+i*n]);
			}
			#endif
		}
		// Update temporal values
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		// Value computation
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;++j){
			if (v1[j]>=v2[j] && v1[j]>=v_explore[j+i*n]){
				value[j+i*n] = v1[j];
				current_value_zone = 1;
			} else if (v2[j]>v1[j] && v2[j]>=v_explore[j+i*n]){
				value[j+i*n] = v2[j];
				current_value_zone = 2;
			} else if (v_explore[j+i*n]>v1[j] && v_explore[j+i*n]>v2[j]){
				value[j+i*n] = v_explore[j+i*n];
				current_value_zone = 0;
			}
			#ifdef DEBUG
			if (j<n-1){
				fprintf(value_file,"%f\t",value[j+i*n]);
			} else {
				fprintf(value_file,"%f\n",value[j+i*n]);
			}
			#endif
			// Bound computation
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j+i*n])<1e-8){
						if (!setted_ub){
							ub[bound_ind] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j+i*n])<1e-8){
						lb[bound_ind] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[bound_ind] = (g[j-1]*(v1[j]-v_explore[j+i*n]) - g[j]*(v1[j-1]-v_explore[j-1+i*n])) / (v_explore[j-1+i*n]-v_explore[j+i*n]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[bound_ind] = lb[bound_ind];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v_explore[j+i*n]-v2[j]) - g[j]*(v_explore[j-1+i*n]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j+i*n]-v_explore[j-1+i*n]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	std::cout<<"Exited backpropagate_value "<<std::endl;
	#endif
	return value[int(0.5*n)];
}
