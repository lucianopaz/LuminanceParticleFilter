#include "DecisionPolicy.hpp"
#include <iostream>
#include <cstdio>

int main(){
	double model_var = 50./0.04;
	double prior_mu_mean = 0.;
	double prior_mu_var = 100;
	int n = 51;
	double dt = 1e-3;
	double T = 10.;
	double reward = 1.;
	double penalty = 0.;
	double iti = 10.;
	double tp = 5.;
	double cost = 100./5.;
	DecisionPolicy dp = DecisionPolicy(model_var, prior_mu_mean, prior_mu_var, n, dt, T,
						reward, penalty, iti, tp, cost);
	int i,j;
	double val;
	FILE* myfile;
	
	//~ val = dp.backpropagate_value(-0.5, false);
	//~ std::cout << "Start value for rho=-0.5" << val << std::endl;
	//~ val = dp.backpropagate_value(-1., false);
	//~ std::cout << "Start value for rho=-1." << val << std::endl;
	//~ val = dp.backpropagate_value(-1.5, false);
	//~ std::cout << "Start value for rho=-1.5" << val << std::endl;
	
	dp.disp();
	dp.iterate_rho_value(1e-12);
	
	std::cout<<dp.backpropagate_value()<<std::endl;
	double* xub = dp.x_ubound();
	double* xlb = dp.x_lbound();
	double g1[dp.nT];
	double g2[dp.nT];
	
	double v1,v2;
	myfile = fopen("v12.txt", "w+");
	for (i=0;i<dp.n;i++){
		v1 = dp.reward*dp.g[i]-dp.penalty*(1.-dp.g[i]) - (dp.iti+(1.-dp.g[i])*dp.tp)*dp.rho;
		v2 = dp.reward*(1.-dp.g[i])-dp.penalty*dp.g[i] - (dp.iti+dp.g[i]*dp.tp)*dp.rho;
		fprintf(myfile,"%f\t%f\n",v1,v2);
	}
	fclose(myfile);
	
	//~ dp.rt(0.1, g1, g2, xub, xlb);
	
	myfile = fopen("test.txt", "w+");
	std::cout<<"Writing to file"<<std::endl;
	for (i=0;i<dp.nT;i++){
		fprintf(myfile,"%f\t%f\t%f\t%f\n",dp.ub[i],dp.lb[i],xub[i],xlb[i]);//,g1[i],g2[i]);
	}
	fclose(myfile);
	
	std::cout<<"Exiting"<<std::endl;
	
	delete[] xub;
	delete[] xlb;
	return 0;
}
