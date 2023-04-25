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


def L96_fun(t,SV,N,F):
    """Equations of the Lorenz 96 model, vectorized"""
    
    dSV_dt = np.array([
        (SV[np.mod(ii+1,N),:] - SV[ii-2,:]) * SV[ii-1,:] - SV[ii,:] + F for ii in range(N)
    ])

    return dSV_dt

def sol_l96(t_span, \
        SV_init, \
        N = 20, F = 8.0, \
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
        L96_fun, \
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


def logL96_fun(t,SV,N,F,ln_vars,rl_vars,xi,scale):
    """Equations of the Lorenz 96 model, vectorized"""
    
    dSV_dt = np.empty_like(SV)
    for ii in range(N):
        x_ip1 = transform_var(SV,np.mod(ii+1,N),ln_vars,rl_vars,xi,scale)
        x_i   = transform_var(SV,ii            ,ln_vars,rl_vars,xi,scale)
        x_im1 = transform_var(SV,ii-1          ,ln_vars,rl_vars,xi,scale)
        x_im2 = transform_var(SV,ii-2          ,ln_vars,rl_vars,xi,scale)

        dSV_dt[ii,:] = (x_ip1 - x_im2) * x_im1 - x_i + F

        if ii in ln_vars:
            dSV_dt[ii,:] = dSV_dt[ii,:]/SV[ii,:]/scale[ii]
        elif ii in rl_vars:
            dSV_dt[ii,:] = dSV_dt[ii,:]/(SV[ii,:]-xi)/scale[ii]
        else:
            dSV_dt[ii,:] = dSV_dt[ii,:]/scale[ii]

    return dSV_dt

def transform_var(SV,ii,ln_vars,rl_vars,xi,scale):
    if ii in ln_vars:
        x = scale[ii]*np.log(SV[ii,:])
    elif ii in rl_vars:
        x = scale[ii]*np.log(xi-SV[ii,:])
    else:
        x = scale[ii]*SV[ii,:]
    return x


def sol_logl96(t_span, \
        SV_init, \
        N = 20, F = 8.0, \
        ln_vars=[], rl_vars=[], xi=1.0, scale=None, \
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

    if scale is None:
        scale = np.ones(N)

    # Solve the Lorenz 63 model
    sol = solve_ivp( \
        logL96_fun, \
        t_span, \
        SV_init, \
        method = meth, \
        t_eval = t_eval, \
        vectorized = True, \
        args = (N, F, ln_vars,rl_vars,xi,scale) \
    )

    # Return none if integration was unsuccessful
    if not sol.success:
        return None, None 

    return sol.t, sol.y
