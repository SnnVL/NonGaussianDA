""" 
Functions needed for solving the Korteweg-de Vries-Burgers model
dx[i]/dt = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F

#### Methods:
- `sol_KdVB`         ->  Solution to the Lorenz-96 model
- `model_KdVB`       ->  Lorenz-96 model for specific use in DA modules

#### Author:
Senne Van Loon
Cooperative Institute for Research in the Atmosphere (CIRA),
Colorado State University,
3925A West Laporte Ave, Fort Collins, CO 80521


"""

import numpy as np
from scipy.integrate import solve_ivp


def KdVB_fun(t,SV,nu,dx,N):
    """Equations of the KdVB model, vectorized"""
    
    dSV_dt = np.array([
        nu/dx/dx*(SV[np.mod(ii+1,N),:] - 2*SV[ii,:] + SV[ii-1,:])\
        -3./dx/dx/dx*(SV[np.mod(ii+2,N),:] - 2*SV[np.mod(ii+1,N),:] + 2*SV[ii-1,:] - SV[ii-2,:])\
        -0.5/dx*SV[ii,:]*(SV[np.mod(ii+1,N),:] - SV[ii-1,:])\
        for ii in range(N)
    ])

    return dSV_dt

def sol_KdVB(t_span, \
        SV_init, \
        nu = 0.07, dx = 0.5, N = 101, \
        t_eval = None, \
        meth = 'RK45'):
    """
    Solution to the Lorenz 96 model 
    
    #### Input
    - `t_span`  ->  Beginning and endpoints in time
    - `SV_init` ->  Initial value for all state variables, array of size n_SV
    - `nu`      ->  Diffusion coefficient
    - `dx`      ->  Grid spacing
    - `N`       ->  Number of state variables
    - `t_eval`  ->  Time points where the model is evaluated
    - `meth`    ->  String indicating the integration method to use

    #### Output
    - `t`       ->  Time vector of size n_t
    - `y`       ->  State variables at every time point, array of size n_SV x n_t
    """

    # Solve the Lorenz 63 model
    sol = solve_ivp( \
        KdVB_fun, \
        t_span, \
        SV_init, \
        method = meth, \
        t_eval = t_eval, \
        vectorized = True, \
        args = (nu,dx,N) \
    )

    # Return none if integration was unsuccessful
    if not sol.success:
        return None, None 

    return sol.t, sol.y
