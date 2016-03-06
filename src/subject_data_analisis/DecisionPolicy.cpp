#include "DecisionPolicy.hpp"

/***
 * DecisionPolicy is a class that implements the dynamic programing method
 * that computes the value of a given belief state as a function of time.
 * 
 * This class implements the method given in Drugowitsch et al 2012 but
 * is limited to constant cost values.
***/

DecisionPolicy::DecisionPolicy(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double cost){
	/***
	 * Constructor
	***/
	int i;
	
	this->model_var = model_var*dt;
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
	for (i=0;i<n;i++){
		this->g[i] = (0.5+double(i))*this->dg;
	}
	this->t = new double[nT];
	for (i=0;i<nT;i++){
		this->t[i] = double(i)*dt;
	}
	this->ub = new double[nT];
	this->lb = new double[nT];
	
	#ifdef DEBUG
	std::cout<<"Created DecisionPolicy instance at "<<this<<std::endl;
	#endif
}

DecisionPolicy::~DecisionPolicy(){
	delete[] g;
	delete[] t;
	delete[] ub;
	delete[] lb;
	#ifdef DEBUG
	std::cout<<"Destroyed DecisionPolicy instance"<<std::endl;
	#endif
}

void DecisionPolicy::disp(){
	std::cout<<"model_var = "<<model_var<<std::endl;
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
	std::cout<<"n = "<<tp<<std::endl;
	std::cout<<"nT = "<<rho<<std::endl;
}

double DecisionPolicy::backpropagate_value(){
	return this->backpropagate_value(this->rho,true);
}

double DecisionPolicy::backpropagate_value(double rho, bool compute_bounds){
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone;
	int current_value_zone;
	int i, j, k, curr_invg, fut_invg;
	double mu_n, post_var_t1, post_var_t, norm_p, maxp;
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
	for (i=0;i<n;i++){
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		value[i] = v1[i]>=v2[i] ? v1[i] : v2[i];
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
				this->lb[nT-1] = g[i];
				this->ub[nT-1] = g[i];
			} else if (i>0 && i<n){
				if ((v1[i]>v2[i] && v1[i-1]<v2[i-1]) || (v1[i]<v2[i] && v1[i-1]>v2[i-1])){
					lb[nT-1] = ((v1[i-1]-v2[i-1])*g[i] + (v1[i]-v2[i])*g[i-1]) / (v2[i]-v1[i]+v1[i-1]-v2[i-1]);
					ub[nT-1] = lb[nT-1];
				}
			}
		}
	}
	post_var_t1 = post_mu_var(this->t[nT-1]);
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		ub[i] = 1.;
		lb[i] = 0.;
		post_var_t = post_mu_var(t[i]);
		for (j=0;j<n;j++){
			v_explore[j] = 0.;
			invg[curr_invg][j] = g2x(t[i],g[j]);
			mu_n = post_mu_mean(t[i],invg[curr_invg][j]);
			norm_p = 0.;
			maxp = -INFINITY;
			for (k=0;k<n;k++){
				p[k] = -0.5*pow(invg[fut_invg][k]-invg[curr_invg][j]-mu_n,2)/(post_var_t+model_var)+
						0.5*pow(invg[fut_invg][k]/model_var+prior_mu_mean/prior_mu_var,2)*post_var_t1;
				maxp = p[k]>maxp ? p[k] : maxp;
				#ifdef DEBUG
				fprintf(details_file,"%f\t%f\t%f\t%f\t%f\n",invg[fut_invg][k],invg[curr_invg][j],mu_n,post_var_t,post_var_t1);
				#endif
			}
			for (k=0;k<n;k++){
				p[k] = exp(p[k]-maxp);
				norm_p+=p[k];
				v_explore[j]+= p[k]*value[k];
			}
			v_explore[j] = v_explore[j]/norm_p - (cost+rho)*dt;
			
			#ifdef DEBUG
			for (k=0;k<n-1;k++){
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
		post_var_t1 = post_var_t;
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;j++){
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
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j])<1e-8){
						if (!setted_ub){
							ub[i] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j])<1e-8){
						lb[i] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[i] = (g[j-1]*(v1[j]-v_explore[j]) - g[j]*(v1[j-1]-v_explore[j-1])) / (v_explore[j-1]-v_explore[j]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[i] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[i] = lb[i];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[i] = (g[j-1]*(v_explore[j]-v2[j]) - g[j]*(v_explore[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j]-v_explore[j-1]);
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

double DecisionPolicy::value_for_root_finding(double rho){
	return this->backpropagate_value(rho,false);
}

double DecisionPolicy::iterate_rho_value(double tolerance){
	//  The Brent algorithm used here to iterate rho's value, was
	//  adapted from brent.cpp written by John Burkardt and Richard
	//  Brent.
	double lower_bound, upper_bound, low_buffer;
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
	lower_bound=-10.; upper_bound = 10.;
	func_at_low_bound = this->value_for_root_finding(lower_bound);
	func_at_up_bound = this->value_for_root_finding(upper_bound);
	if (func_at_low_bound==0){
		this->rho = lower_bound;
	} else if (func_at_up_bound==0){
		this->rho = upper_bound;
	} else {
		// To get a sign changing interval
		while (SIGN(func_at_low_bound)==SIGN(func_at_up_bound)){
			if ((func_at_low_bound<func_at_up_bound and func_at_low_bound<0) ||
			    (func_at_low_bound>func_at_up_bound and func_at_low_bound>0)){
				lower_bound = upper_bound;
				upper_bound*=10;
				func_at_up_bound = this->value_for_root_finding(upper_bound);
			} else if ((func_at_low_bound>func_at_up_bound and func_at_low_bound<0) ||
			    (func_at_low_bound<func_at_up_bound and func_at_low_bound>0)){
				upper_bound = lower_bound;
				lower_bound*=10;
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
	#ifdef DEBUG
	std::cout<<"Entered x_ubound = "<<std::endl;
	#endif
	int i;
	double *xb = new double[nT];
	for (i=0;i<nT;i++){
		xb[i] = g2x(t[i],ub[i]);
	}
	return xb;
}
void DecisionPolicy::x_ubound(double* xb){
	#ifdef DEBUG
	std::cout<<"Entered x_ubound_double* with xb = "<<xb<<std::endl;
	#endif
	int i;
	for (i=0;i<nT;i++){
		xb[i] = g2x(t[i],ub[i]);
	}
}

double* DecisionPolicy::x_lbound(){
	#ifdef DEBUG
	std::cout<<"Entered x_lbound"<<std::endl;
	#endif
	int i;
	double *xb = new double[nT];
	for (i=0;i<nT;i++){
		xb[i] = g2x(t[i],lb[i]);
	}
	return xb;
}
void DecisionPolicy::x_lbound(double* xb){
	#ifdef DEBUG
	std::cout<<"Entered x_lbound_double* with xb = "<<xb<<std::endl;
	#endif
	int i;
	for (i=0;i<nT;i++){
		xb[i] = g2x(t[i],lb[i]);
	}
}

double DecisionPolicy::Psi(double mu, double* bound, int itp, double tp, double x0, double t0){
	double normpdf = 0.3989422804014327*exp(-0.5*pow(bound[itp]-x0-mu*(tp-t0),2)/this->model_var/(tp-t0))/sqrt(this->model_var*(tp-t0));
	double bound_prime = itp<int(sizeof(bound)/sizeof(double)-1) ? (bound[itp+1]-bound[itp])/this->dt : 0.;
	return 0.5*normpdf*(bound_prime-(bound[itp]-x0)/(tp-t0));
}

void DecisionPolicy::rt(double mu, double* g1, double* g2, double* xub, double* xlb){
	#ifdef DEBUG
	std::cout<<"Entered rt"<<std::endl;
	#endif
	unsigned int i,j;
	unsigned int tnT = this->nT;
	double t0,tj,ti;
	bool delete_xub = false;
	bool delete_xlb = false;
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
	g1[1] = -2.*this->Psi(mu,xub,1,this->t[1],this->prior_mu_mean,t0);
	g2[1] = 2.*this->Psi(mu,xlb,1,this->t[1],this->prior_mu_mean,t0);
	for (i=2;i<tnT;i++){
		ti = this->t[i];
		g1[i] = -this->Psi(mu,xub,i,ti,this->prior_mu_mean,t0);
		g2[i] = this->Psi(mu,xlb,i,ti,this->prior_mu_mean,t0);
		for (j=0;j<i;j++){
			tj = this->t[j];
			g1[i]+=this->dt*(g1[j]*this->Psi(mu,xub,i,ti,xub[j],tj)+
								g2[j]*this->Psi(mu,xub,i,ti,xlb[j],tj));
			g2[i]-=this->dt*(g1[j]*this->Psi(mu,xlb,i,ti,xub[j],tj)+
								g2[j]*this->Psi(mu,xlb,i,ti,xlb[j],tj));
		}
		g1[i]*=2.;
		g2[i]*=2.;
	}
	
	if (delete_xub) delete[] xub;
	if (delete_xlb) delete[] xlb;
}
