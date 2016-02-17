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
	this->nT = (int)(T/dt)+1;
	this->cost = cost;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->rho = 0.;
	this->g = new double[n];
	for (i=0;i<n;i++){
		this->g[i] = double(i)/double(n-1);
	}
	this->dg = 1./double(n-1);
	this->t = new double[nT];
	for (i=0;i<nT;i++){
		this->t[i] = double(i)*dt;
	}
	this->ub = new double[nT];
	this->lb = new double[nT];
	
	std::cout<<"Created DecisionPolicy instance"<<std::endl;
}

DecisionPolicy::~DecisionPolicy(){
	delete[] g;
	delete[] t;
	delete[] ub;
	delete[] lb;
	std::cout<<"Destroyed DecisionPolicy instance"<<std::endl;
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

double DecisionPolicy::backpropagate_value(double rho, bool compute_bounds){
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
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
				if ((v1[i]>v2[i] && v1[i-1]<v2[i-1]) || v1[i]<v2[i] && v1[i-1]>v2[i-1]){
					lb[nT-1] = ((v1[i-1]-v2[i-1])*g[i] + (v1[i]-v2[i])*g[i-1]) / (v2[i]-v1[i]+v1[i-1]-v2[i-1]);
					ub[nT-1] = lb[nT-1];
				}
			}
		}
	}
	post_var_t1 = post_mu_var(this->t[nT-1]);
	for (i=nT-2;i>=0;i--){
		if (i%100==0) std::cout<<i<<std::endl;
		setted_ub = false;
		ub[i] = 1.;
		lb[i] = 0.;
		post_var_t = post_mu_var(t[i]);
		for (j=0;j<n;j++){
			invg[curr_invg][j] = g2x(t[i],g[j]);
			mu_n = post_mu_mean(t[i],invg[curr_invg][j]);
			norm_p = 0.;
			maxp = -INFINITY;
			for (k=0;k<n;k++){
				p[k] = -0.5*pow(invg[fut_invg][k]-invg[curr_invg][j]-mu_n,2)/(post_var_t1+model_var)+
						0.5*post_var_t1*pow(invg[fut_invg][k]/model_var+prior_mu_mean/prior_mu_var,2);
				maxp = p[k]>maxp ? p[k] : maxp;
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
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	#endif
	std::cout<<"Exited backpropagate_value "<<std::endl;
	return value[int(0.5*n)];
}

double* DecisionPolicy::x_ubound(){
	std::cout<<"Entered x_ubound = "<<std::endl;
	int i;
	double *xb = new double[nT];
	for (i=0;i<n;i++){
		xb[i] = g2x(t[i],ub[i]);
	}
	return xb;
}

double* DecisionPolicy::x_lbound(){
	std::cout<<"Entered x_lbound = "<<std::endl;
	int i;
	double *xb = new double[nT];
	for (i=0;i<nT;i++){
		xb[i] = g2x(t[i],lb[i]);
	}
	return xb;
}

double DecisionPolicy::Psi(double mu, double* bound, int itp, double tp, double x0, double t0){
	double normpdf = 0.39894228*exp(-0.5*pow(bound[itp]-x0-mu*(tp-t0),2)/this->model_var/(tp-t0))/sqrt(this->model_var*(tp-t0));
	double bound_prime = itp<(sizeof(bound)/sizeof(double)-1) ? (bound[itp+1]-bound[itp])/this->dt : 0.;
	return 0.5*normpdf*(bound_prime-(bound[itp]-x0)/(tp-t0));
}

void DecisionPolicy::rt(double mu, double* g1, double* g2, double* xub, double* xlb){
	int i,j;
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
	
	for (i=0;i<this->nT;i++){
		ti = this->t[i];
		if (i>1){
			g1[i] = -2.*this->Psi(mu,xub,i,ti,this->prior_mu_mean,t0);
			g2[i] = 2.*this->Psi(mu,xlb,i,ti,this->prior_mu_mean,t0);
			for (j=0;j<i;j++){
				tj = this->t[j];
				g1[i]+=2.*this->dt*(g1[j]*this->Psi(mu,xub,i,ti,xub[j],tj)+
									g2[j]*this->Psi(mu,xub,i,ti,xlb[j],tj));
				g2[i]-=2.*this->dt*(g1[j]*this->Psi(mu,xlb,i,ti,xub[j],tj)+
									g2[j]*this->Psi(mu,xlb,i,ti,xlb[j],tj));
			}
		} else if (i==1) {
			g1[i] = -2.*this->Psi(mu,xub,i,ti,this->prior_mu_mean,t0);
			g2[i] = 2.*this->Psi(mu,xlb,i,ti,this->prior_mu_mean,t0);
		} else {
			g1[i] = 0.;
			g2[i] = 0.;
			t0 = ti;
		}
	}
	
	if (delete_xub) delete[] xub;
	if (delete_xlb) delete[] xlb;
}
