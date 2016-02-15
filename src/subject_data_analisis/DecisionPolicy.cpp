#include "DecisionPolicy.hpp"

DecisionPolicy::DecisionPolicy(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double cost){
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
	this->nT = (int)(T/dt);
	this->cost = cost;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->g = new double[n];
	for (i=0;i<n;i++){
		this->g[i] = double(i)/double(n-1);
	}
	this->dg = 1./double(n-1);
	this->t = new double[nT];
	for (i=0;i<nT;i++){
		this->t[i] = double(i)*dt/T;
	}
	this->ub = new double[nT];;
	this->lb = new double[nT];;
}

DecisionPolicy::~DecisionPolicy(){
	delete[] g;
	delete[] t;
	delete[] ub;
	delete[] lb;
}

double DecisionPolicy::backpropagate_value(double rho, bool compute_bounds){
	bool setted_ub = false;
	int previous_value_zone;
	int current_value_zone;
	int i, j, k, curr_invg, fut_invg;
	double mu_n, post_var_t1, post_var_t, norm_p;
	double value[n], v1[n], v2[n], v_explore[n], p[n];
	double invg[2][n];
	
	this->rho = rho;
	for (i=0;i<n;i++){
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		value[i] = v1[i]>=v2[i] ? v1[i] : v2[i];
		invg[1][i] = g2x(t[nT-1],g[i]);
		
		if (compute_bounds){
			if (v1[i]==v2[i]){
				this->lb[nT-1] = g[i];
				this->ub[nT-1] = g[i];
			} else if (i>0 && i<n){
				if (v1[i]>v2[i] && v1[i-1]<v2[i-1]){
					lb[nT-1] = ((v1[i-1]-v2[i-1])*g[i] + (v1[i]-v2[i])*g[i-1]) / (v2[i]-v1[i]+v1[i-1]-v2[i-1]);
					ub[nT-1] = lb[nT-1];
				}
			}
		}
	}
	curr_invg = 0;
	post_var_t1 = post_mu_var(this->t[nT-1]);
	for (i=nT-2;i>=0;i--){
		setted_ub = false;
		ub[i] = 1.;
		lb[i] = 0.;
		post_var_t = post_mu_var(t[i]);
		for (j=0;j<n;j++){
			invg[curr_invg][j] = g2x(t[i],g[j]);
			mu_n = post_mu_mean(t[i],invg[curr_invg][j]);
			norm_p = 0.;
			for (k=0;k<n;k++){
				p[k] = -0.5*pow(invg[fut_invg][k]-invg[curr_invg][j]-mu_n,2)/(post_var_t1+model_var)+
						0.5*post_var_t1*pow(invg[fut_invg][k]/model_var+prior_mu_mean/prior_mu_var,2);
				norm_p+=p[k];
				v_explore[j]+= p[k]*value[k];
			}
			v_explore[j] = v_explore[j]/norm_p - (cost+rho)*dt;
		}
		post_var_t1 = post_var_t;
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
			if (compute_bounds){
				if (i>0 && i<n){
					if (v1[j]==v_explore[j]){
						if (!setted_ub){
							ub[i] = g[j];
							setted_ub = true;
						}
					} else if (v2[j]==v_explore[j]){
						lb[i] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && !setted_ub){
							ub[i] = (g[i-1]*(v1[i]-v_explore[i]) - g[i]*(v1[i-1]-v_explore[i-1])) / (v_explore[i-1]-v_explore[i]+v1[i]-v1[i-1]);
						} else if (current_value_zone==0){
							lb[i] = (g[i-1]*(v_explore[i]-v2[i]) - g[i]*(v_explore[i-1]-v2[i-1])) / (v2[i-1]-v2[i]+v_explore[i]-v_explore[i-1]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	return value[int(0.5*n)];
}

double* DecisionPolicy::x_ubound(){
	int i;
	double *xb = new double[nT];
	for (i=0;i<n;i++){
		xb[i] = g2x(t[i],ub[i]);
	}
	return xb;
}

double* DecisionPolicy::x_lbound(){
	int i;
	double *xb = new double[nT];
	for (i=0;i<nT;i++){
		xb[i] = g2x(t[i],lb[i]);
	}
	return xb;
}


	//~ def rt(self,mu,bounds=None):
		//~ if bounds is None:
			//~ bounds = self.belief_bound_to_x_bound(self.bounds)
		//~ g1 = np.zeros_like(self.t)
		//~ g2 = np.zeros_like(self.t)
		//~ 
		//~ for i,t in enumerate(self.t):
			//~ if i==0:
				//~ t0 = t
				//~ continue
			//~ elif i==1:
				//~ g1[i] = -2*self.Psi(mu,bounds[0],i,t,self.prior_mu_mean,t0)
				//~ g2[i] = 2*self.Psi(mu,bounds[1],i,t,self.prior_mu_mean,t0)
			//~ else:
				//~ g1[i] = -2*self.Psi(mu,bounds[0],i,t,self.prior_mu_mean,t0)+\
					//~ 2*self.dt*np.sum(g1[:i]*self.Psi(mu,bounds[0],i,t,bounds[0][:i],self.t[:i]))+\
					//~ 2*self.dt*np.sum(g2[:i]*self.Psi(mu,bounds[0],i,t,bounds[1][:i],self.t[:i]))
				//~ g2[i] = 2*self.Psi(mu,bounds[1],i,t,self.prior_mu_mean,t0)-\
					//~ 2*self.dt*np.sum(g1[:i]*self.Psi(mu,bounds[1],i,t,bounds[0][:i],self.t[:i]))-\
					//~ 2*self.dt*np.sum(g2[:i]*self.Psi(mu,bounds[1],i,t,bounds[1][:i],self.t[:i]))
		//~ return g1,g2
		//~ 
	//~ def Psi(self,mu,bound,itp,tp,x0,t0):
		//~ normpdf = np.exp(-0.5*(bound[itp]-x0-mu*(tp-t0))**2/self.model_var/(tp-t0))/np.sqrt(2.*np.math.pi*self.model_var*(tp-t0))
		//~ bound_prime = (bound[itp+1]-bound[itp])/self.dt if itp<len(bound)-1 else 0.
		//~ return 0.5*normpdf*(bound_prime-(bound[itp]-x0)/(tp-t0))
