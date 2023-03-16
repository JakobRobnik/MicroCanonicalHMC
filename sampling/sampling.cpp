#include<iostream>
#include <random>
using namespace std;



// Some convenient functions

double norm(double *vector, int len){
    //norm of the vector
	double S = 0.0;
	for(int i = 0; i < len; i++)S += vector[i]*vector[i];
	return sqrt(S);
}

double dot(double *a, double *b, int len){
    //dot product
	double S = 0.0;
	for(int i = 0; i < len; i++)S += a[i]*b[i];
	return S;
}

double *empty(int len){
	//reserves an array of doubles of length len
	double *array = (double *)malloc(len*sizeof(double));
	return array;
}

double **empty_matrix(int num_samples, int d){
	double **M = (double **)malloc(num_samples*sizeof(double *));
	for(int i = 0; i< num_samples; ++i)M[i] = (double *)malloc(d*sizeof(double));
	return M;
}


std::normal_distribution<> randn(0.0, 1.0); // create normal distribution




// The target distribution is defined as a class with atributes:
//    d: dimension
//    grad_nlogp: a function which updates the -log p and its gradient for the new x.
//    prior_draw: random draw from a prior, used to initialize the sampler


class Target {
  // target distribution that we want to sample from
  public:
    int d; //configuration space dimension
    void grad_nlogp(double *, double *, double *); // takes the position x and the pointers where -nlogp and its gradient should be stored.
    double *prior_draw(std::mt19937); //random draw from a prior, used to initialize the sampler

};

// an example of a target distribution: standard normal

inline void Target::grad_nlogp(double *x, double *l, double *g){
    double S = 0.0;
    for(int i = 0; i< d; ++i){
        S += + pow(x[i], 2); // - log p(x) = 0.5 \sum x_i^2
        g[i] = x[i]; // grad (-log p) = x
    }
    l[0] = 0.5 * S;
}

inline double *Target::prior_draw(std::mt19937 gen){
    double *x = (double *)malloc(d*sizeof(double));
    for(int i = 0; i< d; i++)x[i] = 3 * randn(gen); // Gaussian which is broader than posterior
    return x;
}




class Sampler{
    // Sequential MCHMC sampler
    public:
        int num_samples;
        double **samples;
        double *E;
        double *nlogp;

        int burnin;
        double varE;

        double L; double stepsize;

        std::mt19937 gen;

        Target target;

        Sampler(Target _target, double _L, double _eps, std::mt19937 _gen){
            target = _target;
            L = _L; stepsize = _eps;
            gen = _gen;
        }

        // Random generators

        double *random_unit_vector(void){

            double *u = (double *)malloc(target.d*sizeof(double));
            for(int i = 0; i< target.d; i++)u[i] = randn(gen);
            double u_norm = norm(u, target.d);
            for(int i = 0; i< target.d; i++)u[i] = u[i]/u_norm;
            return u;
        }

        void partially_refresh_momentum(double *u, double nu){

            for(int i = 0; i< target.d; i++)u[i] += nu * randn(gen); //add random noise
            double u_norm = norm(u, target.d); //normalize
            for(int i = 0; i< target.d; i++)u[i] = u[i]/u_norm;
        }


        // Hamiltonian dynamics

        double update_momentum(double eps, int d, double *g, double *u){
            //The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
            //similar to the implementation: https://github.com/gregversteeg/esh_dynamics
            //There are no exponentials e^delta, which prevents overflows when the gradient norm is large.

            double g_norm = norm(g, d);

            //update u
            double ue = -dot(g, u, d) / g_norm;
            double delta = eps * g_norm / (d-1);
            double zeta = exp(-delta);
            for(int i = 0; i < d; ++i)u[i] = (-g[i]/g_norm) *(1-zeta)*(1+zeta + ue * (1-zeta)) + 2*zeta* u[i];

            //normalize u
            double u_norm = norm(u, d);
            for(int i = 0; i < d; ++i)u[i] /= u_norm;

            //return the change in the kinetic energy
            return delta - log(2) + log(1 + ue + (1-ue)*zeta*zeta);
        }


        double leapfrog(double eps, double *x, double *u, double *l, double *g){
            //leapfrog integrator

            //half step in momentum
            double kinetic1 = update_momentum(eps * 0.5, target.d, g, u);

            //full step in x
            for(int i = 0; i< target.d; i++)x[i] += eps * u[i];

            target.grad_nlogp(x, l, g);

            //half step in momentum
            double kinetic2 = update_momentum(eps * 0.5, target.d, g, u);

            return (kinetic1 + kinetic2) * (target.d-1);
        }


        double dynamics(double eps, double nu, double *x, double *u, double *l, double *g){
            // One step of the Langevin-like dynamics.

            // Hamiltonian step
            double kinetic = leapfrog(eps, x, u, l, g);

            // add noise to the momentum direction
            partially_refresh_momentum(u, nu);

            return kinetic;
        }


        // sampling

        void sample(int n){
            // Do MCHMC sampling for n steps with optionally thinned samples

            // allocate space for results
            num_samples = n;
            samples = empty_matrix(num_samples, target.d);
            E = empty(num_samples);
            nlogp = empty(num_samples);

            // initialize the particle
            double *x = target.prior_draw(gen);
            double *u = random_unit_vector();

//            for(int i =0; i< target.d; ++i){
//                u[i] = 0.0;
//                x[i] = 1.0/sqrt(target.d);
//            }
//            u[0] = 1.0;

            double *l = empty(1);
            double *g = empty(target.d);
            target.grad_nlogp(x, l, g);
            nlogp[0] = l[0]; E[0] = 0.0;
            for(int id = 0; id < target.d; ++id)samples[0][id] = x[id];

            double nu = sqrt((exp(2 * stepsize / L) - 1.0) / target.d);
            double kinetic_change;
            // do the sampling

            for(int isample = 1; isample< num_samples; ++isample){
               kinetic_change = dynamics(stepsize, nu, x, u, l, g);
               nlogp[isample] = l[0];
               E[isample] = E[isample-1] + kinetic_change + nlogp[isample] - nlogp[isample-1];

               for(int id = 0; id < target.d; ++id)samples[isample][id] = x[id];

            }

            free(x); free(u); free(g);

            //determine the end of the burn in and the variance of the energy
            burnin = burn_in_ending();
            double E1 = 0.0; double E2 = 0.0;
            for(int i = burnin; i < num_samples; ++i){
                E1 += E[i]; E2 += pow(E[i], 2);
            }
            E1 /= (num_samples-burnin); E2 /= (num_samples-burnin);
            varE = (E2 - pow(E1, 2)) / target.d;

        }

        int burn_in_ending(void){
            // Estimate the index at which the burn-in ends
            double loss_avg = 0.0;
            for(int i = 0; i < num_samples; ++i)loss_avg += nlogp[i];
            loss_avg = loss_avg / num_samples;

            int i = 0;
            while((nlogp[i] > loss_avg) & (i < num_samples))++i;
            return i;
        }
};



int main(void){

    std::random_device rd;
    std::mt19937 gen(rd()); // create and seed the generator

    Target target;
    target.d = 100;

    Sampler sampler = Sampler(target, sqrt(target.d), 0.7 * sqrt(target.d), gen);

    sampler.sample(10000); // do the sampling


    printf("Burn in ended after %d steps. Var[E]/d = %lf.\n", sampler.burnin, sampler.varE);

    double S;

    printf("\nExpectation values:\n");
    for(int j = 0; j<10; j++){
        S = 0.0;
        for(int i = sampler.burnin; i< sampler.num_samples; ++i)S += pow(sampler.samples[i][j], 2);
        S /= (sampler.num_samples-sampler.burnin);

        printf("<x%d^2> = %lf\n", j+1, S);
    }

	return 0;
}


