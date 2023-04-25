// Math functions
#define _USE_MATH_DEFINES
#include <math.h>

// cc -fPIC -shared -o C_Lorenz.so models_rk4.c 


// Parameters for different models
struct params_cL63 {int n; double s; double r; double b; double cx; double cy; double cz; };
struct params_gL63 {int M; double s; double r; double a2; };
struct params_L63 {double s; double r; double b; };
struct params_L96 {int N; double F; };
struct params_L05 {int N; double F; int K; };


// Runge-Kutta method of order 4
int rk4_step(int (*f) (double, double*, double*, void*),
    double dt, 
    double t, 
    double* x, 
    double* x_next,
    int N, 
    void *p) {

    double k1[N], k2[N], k3[N], k4[N], x_tmp[N] ;
    int ii ; 

    f(t, x, k1, p) ;
    for (ii=0; ii < N; ii++) {
        x_tmp[ii] = x[ii] + dt*k1[ii]/2.0 ;
    }
    f(t + dt/2.0, x_tmp, k2, p) ;
    for (ii=0; ii < N; ii++) {
        x_tmp[ii] = x[ii] + dt*k2[ii]/2.0 ;
    }
    f(t + dt/2.0, x_tmp, k3, p) ;
    for (ii=0; ii < N; ii++) {
        x_tmp[ii] = x[ii] + dt*k3[ii] ;
    }
    f(t + dt, x_tmp, k4, p) ;
    for (ii=0; ii < N; ii++) {
        x_next[ii] = x[ii] + dt*(k1[ii] + 2.0*k2[ii] + 2.0*k3[ii] + k4[ii])/6.0 ;
    }

	return 0 ;
}

int rk4_sol(int (*f) (double, double*, double*, void*), 
    double t0, 
    double dt, 
    int nt,
    double* x0, 
    double* sol,
    int N, 
    void *p) {

    // Initialize solution and temporary variables
    double current[N], next[N] ;
    double time ;

    // Initial conditions
    for (int ii = 0; ii < N; ii++){
        sol[0 + ii] = x0[ii] ;
    }

    // Loop in time
    for (int jj = 0; jj < nt-1; jj++){
        // Current time
        time = t0+jj*dt ;

        // Current x
        for (int ii = 0; ii < N; ii++){
            current[ii] = sol[jj*N + ii] ;
        }

        // Perform forward step according to RK4
        rk4_step(f, dt, time, current, next, N, p) ;

        // Save x
        for (int ii = 0; ii < N; ii++){
            sol[(jj+1)*N + ii] = next[ii] ;
        }
    }

    return 0 ;
}

int cyc_ind(int ii, int N) {

    if (ii < 0) {
        ii = ii + N ;
    } else if (ii >= N) {
        ii = ii - N ;
    }

    return ii ;
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////       Lorenz-63        ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int lorenz63_fun(double t, double *x, double *dxdt, void *p) {
    // Lorenz, E. N. (1963). Deterministic nonperiodic flow. 
    // Journal of atmospheric sciences, 20(2), 130-141.

    // Pass variables to the function
    struct params_L63 * params = (struct params_L63 *)p;
    double s = (params->s);
    double r = (params->r);
    double b = (params->b);

    dxdt[0] = -s*(x[0]-x[1]) ;
    dxdt[1] = r*x[0]-x[1]-x[0]*x[2] ;
    dxdt[2] = x[0]*x[1]-b*x[2] ;

    return 0;
}

int sol_L63_(double t0, double dt, int nt,
    double *x0, double *sol,
    double s, double r, double b) {

    int N = 3 ;

    // Initialize parameters
    struct params_L63 p = {s, r, b };

    rk4_sol(lorenz63_fun, t0, dt, nt, x0, sol, N, &p) ;

    return 0 ;

}




///////////////////////////////////////////////////////////////////////////////
///////////////////////       Coupled Lorenz-63        ////////////////////////
///////////////////////////////////////////////////////////////////////////////

int coupled_lorenz63_fun(double t, double *x, double *dxdt, void *p) {

    // Pass variables to the function
    struct params_cL63 * params = (struct params_cL63 *)p;
    int    n  = (params->n );
    double s  = (params->s );
    double r  = (params->r );
    double b  = (params->b );
    double cx = (params->cx);
    double cy = (params->cy);
    double cz = (params->cz);

    double lx, ly, lz ;
    for (int ii = 0; ii < n; ii++) {
        lx = x[0+3*ii] ;
        ly = x[1+3*ii] ;
        lz = x[2+3*ii] ;

        dxdt[0+3*ii] = -s*(lx-ly)    + cx * x[cyc_ind(0+3*ii+3,3*n)] ;
        dxdt[1+3*ii] = r*lx-ly-lz*lx + cy * x[cyc_ind(1+3*ii+3,3*n)] ;
        dxdt[2+3*ii] = lx*ly-b*lz    + cz * x[cyc_ind(2+3*ii+3,3*n)] ;
    }

    return 0;
}

int sol_cL63_(double t0, double dt, int nt,
    double *x0, double *sol, int n,
    double s, double r, double b,
    double cx, double cy, double cz) {

    // Initialize parameters
    struct params_cL63 p = {n, s, r, b, cx, cy, cz };

    rk4_sol(coupled_lorenz63_fun, t0, dt, nt, x0, sol, 3*n, &p) ;

    return 0 ;

}



///////////////////////////////////////////////////////////////////////////////
///////////////////////////       Lorenz-96        ////////////////////////////
///////////////////////////////////////////////////////////////////////////////


int lorenz96_fun(double t, double *x, double *dxdt, void *p) {
    // Lorenz, E. N. (1996, September). Predictability: A problem partly solved. 
    // In Proc. Seminar on predictability (Vol. 1, No. 1).

    // Pass variables to the function
    struct params_L96 * params = (struct params_L96 *)p;
    int N = (params->N);
    double F = (params->F);

    for(int ii = 0; ii < N; ii++){
        dxdt[ii] = (x[cyc_ind(ii+1,N)] - x[cyc_ind(ii-2,N)]) * x[cyc_ind(ii-1,N)]
                    - x[ii] + F ;
    }

    return 0;
}

int sol_L96_(double t0, double dt, int nt,
    double *x0, double *sol,
    int N, double F) {

    // Initialize parameters
    struct params_L96 p = {N, F };
    rk4_sol(lorenz96_fun, t0, dt, nt, x0, sol, N, &p) ;

    return 0 ;

}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////       Lorenz-05        ////////////////////////////
///////////////////////////////////////////////////////////////////////////////


int lorenz05_fun(double t, double *x, double *dxdt, void *p) {
    // Lorenz, E. N. (2005). Designing chaotic models. 
    // Journal of the atmospheric sciences, 62(5), 1574-1587.

    // Pass variables to the function
    struct params_L05 * params = (struct params_L05 *)p;
    int N = (params->N);
    double F = (params->F);
    int K = (params->K);

    double Wn[N] ;
    int J ;
    int ii, jj ;

    if (K % 2 == 0) {
        // Even K
        J = K/2 ;

        for (ii = 0; ii < N; ii++){
            Wn[ii] = 0.0 ;
            for (jj = -J+1; jj < J; jj++) {
                Wn[ii] = Wn[ii] + x[cyc_ind(ii-jj,N)]/K ;
            }
            Wn[ii] = Wn[ii] + 0.5*x[cyc_ind(ii+J,N)]/K ;
            Wn[ii] = Wn[ii] + 0.5*x[cyc_ind(ii-J,N)]/K ;
        }
        for (ii = 0; ii < N; ii++){
            dxdt[ii] = -Wn[cyc_ind(ii-2*K,N)]*Wn[cyc_ind(ii-K,N)] - x[ii] + F ;
            for (jj = -J+1; jj < J; jj++) {
                dxdt[ii] = dxdt[ii] + Wn[cyc_ind(ii-K+jj,N)]*x[cyc_ind(ii+K+jj,N)]/K ;
            }
            dxdt[ii] = dxdt[ii] + 0.5*Wn[cyc_ind(ii-K+J,N)]*x[cyc_ind(ii+K+J,N)]/K ;
            dxdt[ii] = dxdt[ii] + 0.5*Wn[cyc_ind(ii-K-J,N)]*x[cyc_ind(ii+K-J,N)]/K ;
        }
    } else {
        // Odd K
        J = (K-1)/2 ;

        for (ii = 0; ii < N; ii++){
            Wn[ii] = 0.0 ;
            for (jj = -J; jj < J+1; jj++) {
                Wn[ii] = Wn[ii] + x[cyc_ind(ii-jj,N)]/K ;
            }
        }
        for (ii = 0; ii < N; ii++){
            dxdt[ii] = -Wn[cyc_ind(ii-2*K,N)]*Wn[cyc_ind(ii-K,N)] - x[ii] + F ;
            for (jj = -J; jj < J+1; jj++) {
                dxdt[ii] = dxdt[ii] + Wn[cyc_ind(ii-K+jj,N)]*x[cyc_ind(ii+K+jj,N)]/K ;
            }
        }
    }

    return 0;
}

int sol_L05_(double t0, double dt, int nt,
    double *x0, double *sol,
    int N, double F, int K) {

    // Initialize parameters
    struct params_L05 p = {N, F, K };
    rk4_sol(lorenz05_fun, t0, dt, nt, x0, sol, N, &p) ;

    return 0 ;

}


///////////////////////////////////////////////////////////////////////////////
/////////////////////       Generalized Lorenz-63        //////////////////////
///////////////////////////////////////////////////////////////////////////////

int generalized_lorenz63_fun(double t, double *x, double *dxdt, void *p) {
    // Shen, B. W. (2019). Aggregated negative feedback in a generalized Lorenz model. 
    // International Journal of Bifurcation and Chaos, 29(03), 1950037.

    // Pass variables to the function
    struct params_gL63 * params = (struct params_gL63 *)p;
    int M = (params->M);
    double s = (params->s);
    double r = (params->r);
    double a2 = (params->a2);

    int N = (M-3)/2 ;
    double b = 4.0/(1.0+a2) ;
    double d, beta ;
    double yj, yjp1, zj, zjm1;

    // First three variables
    dxdt[0] = -s*(x[0]-x[1]) ;
    dxdt[1] = r*x[0]-x[1]-x[0]*x[2] ;
    dxdt[2] = x[0]*x[1]-x[0]*x[3]-b*x[2] ;

    for(int jj = 1; jj <= N; jj++){

        // Parameters
        d = ((2.0*jj+1.0)*(2.0*jj+1.0)+a2)/(1.0+a2) ;
        beta = (jj+1.0)*(jj+1.0)*b ;

        // Variables
        yj = x[2  +jj];
        zj = x[2+N+jj];
        if (jj==1) {
            yjp1 = x[2+jj+1];
            zjm1 = x[2];
        } else if (jj==N) {
            yjp1 = 0.0;
            zjm1 = x[2+N+jj-1];
        } else {
            yjp1 = x[2+  jj+1];
            zjm1 = x[2+N+jj-1];
        }

        // Derivatives
        dxdt[2  +jj] = jj*x[0]*zjm1-(jj+1.0)*x[0]*zj-d*yj;
        dxdt[2+N+jj] = (jj+1.0)*x[0]*yj-(jj+1.0)*x[0]*yjp1-beta*zj;
    }

    return 0;
}

int sol_gL63_(double t0, double dt, int nt,
    double *x0, double *sol,
    int M, double s, double r, double a2) {

    // Initialize parameters
    struct params_gL63 p = {M, s, r, a2 };

    rk4_sol(generalized_lorenz63_fun, t0, dt, nt, x0, sol, M, &p) ;

    return 0 ;

}