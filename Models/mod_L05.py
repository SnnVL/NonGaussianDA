""" 
Functions needed for solving the Lorenz 96 model
dx[i]/dt = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F

#### Methods:
- `sol_l96`         ->  Solution to the Lorenz-96 model
- `model_l96`       ->  Lorenz-96 model for specific use in DA modules

#### Author:
Senne Van Loon
Cooperative Institute for Research in the Atmosphere (CIRA),
Colorado State University,
3925A West Laporte Ave, Fort Collins, CO 80521


"""

import numpy as np
from scipy.integrate import solve_ivp


def model1_fun(t,SV,N,F):
    """Equations of the Lorenz 96 model, vectorized"""
    
    dSV_dt = np.array([
        (SV[np.mod(ii+1,N),:] - SV[ii-2,:]) * SV[ii-1,:] - SV[ii,:] + F for ii in range(N)
    ])

    return dSV_dt

def sol_model1(t_span, \
        SV_init, \
        N = 30, F = 8.0, \
        t_eval = None, \
        meth = 'RK45'):
    """
    Solution to the Lorenz 96 model 
    
    #### Input
    - `t_span`  ->  Beginning and endpoints in time
    - `SV_init` ->  Initial value for all state variables, array of size n_SV
    - `N`       ->  Number of state variables
    - `F`       ->  Constant forcing
    - `t_eval`  ->  Time points where the model is evaluated
    - `meth`    ->  String indicating the integration method to use

    #### Output
    - `t`       ->  Time vector of size n_t
    - `y`       ->  State variables at every time point, array of size n_SV x n_t
    """

    # Solve the Lorenz 63 model
    sol = solve_ivp( \
        model1_fun, \
        t_span, \
        SV_init, \
        method = meth, \
        t_eval = t_eval, \
        vectorized = True, \
        args = (N, F) \
    )

    # Return none if integration was unsuccessful
    if not sol.success:
        return None, None 

    return sol.t, sol.y


################################################################################
################################################################################
################################################################################

def model2_fun(t,SV,N,F,K):
    """Equations of the Lorenz 96 model, vectorized"""
    
    if np.mod(K,2) == 0:
        # Even K
        J = int(K/2)
        Wn = np.array([
            np.sum(SV[np.mod(np.arange(n-J+1,n+J),N),:], axis=0) \
            + (SV[n-J,:]+SV[np.mod(n+J,N),:])/2.0 \
            for n in range(N)
        ])/K
        dSV_dt = np.array([
            -Wn[n-2*K,:]*Wn[n-K,:] \
            + Wn[n-K-J,:]*SV[np.mod(n+K-J,N),:]/2.0/K \
            + Wn[n-K+J,:]*SV[np.mod(n+K+J,N),:]/2.0/K \
            + np.sum(Wn[np.mod(np.arange(n-K-J+1,n-K+J),N),:] \
                *SV[np.mod(np.arange(n+K-J+1,n+K+J),N),:], axis=0)/K \
            - SV[n,:] + F \
            for n in range(N)
        ])
    else:
        # Odd K
        J = int((K-1)/2)
        Wn = np.array([
            np.sum(SV[np.mod(np.arange(n-J,n+J+1),N),:], axis=0) for n in range(N)
        ])/K
        dSV_dt = np.array([
            -Wn[n-2*K,:]*Wn[n-K,:] \
            + np.sum(Wn[np.mod(np.arange(n-K-J,n-K+J+1),N),:] \
                *SV[np.mod(np.arange(n+K-J,n+K+J+1),N),:], axis=0)/K \
            - SV[n,:] + F \
            for n in range(N)
        ])

    return dSV_dt

def sol_model2(t_span, \
        SV_init, \
        N = 240, F = 15.0, K = 8.0, \
        t_eval = None, \
        meth = 'RK45'):
    """
    Solution to the Lorenz 96 model 
    
    #### Input
    - `t_span`  ->  Beginning and endpoints in time
    - `SV_init` ->  Initial value for all state variables, array of size n_SV
    - `N`       ->  Number of state variables
    - `F`       ->  Constant forcing
    - `t_eval`  ->  Time points where the model is evaluated
    - `meth`    ->  String indicating the integration method to use

    #### Output
    - `t`       ->  Time vector of size n_t
    - `y`       ->  State variables at every time point, array of size n_SV x n_t
    """

    # Solve the Lorenz 63 model
    sol = solve_ivp( \
        model2_fun, \
        t_span, \
        SV_init, \
        method = meth, \
        t_eval = t_eval, \
        vectorized = True, \
        args = (N, F, K) \
    )

    # Return none if integration was unsuccessful
    if not sol.success:
        return None, None 

    return sol.t, sol.y

