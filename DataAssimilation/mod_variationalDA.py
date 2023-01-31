""" 
Functions needed for variational data assimilation techniques

#### Methods:
- `var3d`   ->  3DVAR DA technique
- `var4d`   ->  4DVAR DA technique

#### Author:
Senne Van Loon
Cooperative Institute for Research in the Atmosphere (CIRA),
Colorado State University,
3925A West Laporte Ave, Fort Collins, CO 80521

#### References and acknowledgements:
[1] Fletcher, S. J. (2017). Data assimilation for the geosciences: From theory to application. Elsevier
[2] Goodliff, M., Fletcher, S., Kliewer, A., Forsythe, J., & Jones, A. (2020). Detection of non-Gaussian behavior using machine learning techniques: A case study on the Lorenz 63 model. Journal of Geophysical Research: Atmospheres, 125, e2019JD031551. https://doi.org/10.1029/2019JD031551
[3] Goodliff, M. R., Fletcher, S. J., Kliewer, A. J., Jones, A. S., & Forsythe, J. M. (2022). Non-Gaussian detection using machine learning with data assimilation applications. Earth and Space Science, 9, e2021EA001908. https://doi.org/10.1029/2021EA001908
[4] Fletcher, S. J. and Jones, A. S., 2014. Multiplicative and additive incremental variational data assimilation for mixed lognormal-Gaussian errors. Monthly Weather Review, 142(7), pp.2521-2544.
[5] Fletcher, S. J., 2010. Mixed Gaussian-lognormal four-dimensional data assimilation. Tellus A: Dynamic Meteorology and Oceanography, 62(3), pp.266-287.
[6] Based on code from Goodliff, M. used in [2] and [3]
[7] Based on code from Hossen, J.

"""

# Load modules
import numpy as np
from scipy.optimize import minimize, root
from scipy.linalg import pinv
import copy

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

from mod_DA_general import getBupdate, get_ln_rl_var, transform_vars, time_limit, TimeoutException


def var3d(init_guess, t_obs, n_t_mod, y, H, B, R, model, \
    ln_vars_SV = [], ln_vars_obs = [], \
    rl_vars_SV = [], rl_vars_obs = [], xi_SV = 0.0, xi_obs = 0.0, method = 'root', l_SV = 1.0, l_obs = 1.0):
    """
    3DVAR

    3DVAR data assimilation technique for use in a general model, which allows for 
    Gaussian and mixed Gaussian-lognormal observations and background state variables

    #### Input
    - `init_guess`  ->  Initial guess for the model analysis, vector of size n_SV
    - `t_obs`       ->  Time values of observations, vector of size n_t_obs
    - `n_t_mod`     ->  Number of time steps to use in each model run
    - `y`           ->  Observations, array of size n_obs x n_t_obs
    - `H`           ->  Observation operator, obtained from create_H, 
                        array of size n_obs x n_SV
    - `B`           ->  Background error covariance matrix, array of size n_SV x n_SV
    - `R`           ->  Observation error covariance matrix, array of size n_obs x n_obs
                        This matrix is assumed to be constant throughout the assimilation
    - `model`       ->  Model to use in the analysis, function of the form
                            x, M = model(t,x_0),
                        where t contains the time values where the model is evaluated
                        and x_0 is the initial condition;
                        the output x are the state variables for all time values
    - `ln_vars_SV`  ->  State variables that should be treated lognormally, 
                        * If ln_vars is a callable, it should be of the form
                                ln_var = ln_vars(SV_input),
                          where SV_input contains the state variables at one time step
                          and ln_var is a list of indices between 0 and n_SV-1
                        * If ln_vars is a list of indices between 0 and n_SV-1,
                          the state variables of these indices are treated as 
                          lognormally distributed for all time steps
                        * If ln_vars is a list of lists, the length of the top list
                          should be n_t_obs, and the inner lists are indices between 0 and n_SV-1,
                          the state variables of these indices are treated as 
                          lognormally distributed for each specific time step
                        Default is [] for all Gaussian variables
    - `ln_vars_obs` ->  Observations that should be treated lognormally, 
                        * If ln_vars is a callable, it should be of the form
                                ln_var = ln_vars(y_input),
                          where y_input contains the observed variables at one time step
                          and ln_var is a list of indices between 0 and n_obs-1
                        * If ln_vars is a list of indices between 0 and n_obs-1,
                          the observations of these indices are treated as 
                          lognormally distributed for all time steps
                        * If ln_vars is a list of lists, the length of the top list
                          should be n_t_obs, and the inner lists are indices between 0 and n_obs-1,
                          the observations of these indices are treated as 
                          lognormally distributed for each specific time step
                        Default is [] for all Gaussian variables
    - `rl_vars_SV`  ->  State variables that should be treated reverse lognormally, same as ln_vars_SV
    - `rl_vars_obs` ->  Observations that should be treated reverse lognormally, same as ln_vars_obs
    - `xi_SV`       ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`      ->  Parameter for the reverse lognormal distribution for the observations
    - `method`      ->  Method for analysis, either `root` (default) for finding the root of the gradient,
                        or `min` for finding the minimum of the cost function
    - `l_SV`        ->  Coefficient to use in the cost function of the background
                        to select the descriptor to optimize
                            l_SV  =  1.0    ->  mode
                            l_SV  = -0.5    ->  mean
                            l_SV  =  0.0    ->  median
    - `l_obs`       ->  Coefficient to use in the cost function of the observations
                        to select the descriptor to optimize
                            l_obs =  1.0    ->  mode
                            l_obs = -0.5    ->  mean
                            l_obs =  0.0    ->  median

    #### Output
    - `x_b`         ->  Background state, array of size n_SV x n_t
    - `x_a`         ->  Analysis state, array of size n_SV x n_t_obs
    - `t_true`      ->  Total time for background, vector of size n_t
    """
    
    # Save lengths of arrays
    n_SV = init_guess.size      # number of state variables
    n_t_obs = t_obs.size        # number of observations
    n_t = n_t_obs * n_t_mod + 1 # number of total time steps

    # Initialize background and analysis states
    x_b = np.empty((n_SV,n_t))
    x_a = np.empty((n_SV,n_t_obs))

    # At t=0, background and analysis states are set to the initial guess
    x_b[:,0] = init_guess
    x_a[:,0] = init_guess
    x_e = init_guess

    # Total time evolution for model
    dt_obs = t_obs[1] - t_obs[0] # Assuming evenly spaced observations
    t_true = np.linspace(t_obs[0],t_obs[-1] + dt_obs, n_t) # Assuming one prediction window

    # Invert observation error covariance matrix
    Rinv = pinv(R)

    timeout = False
    # Loop over all observations
    for ii in range(n_t_obs - 1):
        # Calculated states are on model time, and not observation time
        tt = ii * n_t_mod       # Index for model time at observation point
        tt_next = tt + n_t_mod  # Index for model time at next observation point
        
        # Run the model for the analysis and background
        sim_time = t_true[tt:tt_next+1]
        try:
            with time_limit(60):
                x_a_model = model(sim_time, x_a[:,ii]) # Model run for analysis state
                x_b_model = model(sim_time, x_e)       # Model run for background state (needed for B update)

        except TimeoutException:
            print("Warning: timeout in model run, skipping this assimilation step!")
            print("    x_a_init = "+str(x_a[:,ii]))
            print("    x_b_init = "+str(x_b[:,tt]))

            # If timeout, something went wrong in previous analysis
            x_a[:,ii] = np.nan*np.empty(n_SV)
            x_b[:,tt:tt_next+1] = np.nan*np.empty_like(sim_time)

            # Restart DA from initial guess next step
            x_a[:,ii+1] = init_guess
            x_e = init_guess
            timeout = True
            continue

        # Save model run in background
        x_b[:,tt+1:tt_next+1] = x_a_model[:,1:]
        # x_e = x_a_model[:,-1]
        x_e = x_b_model[:,-1]

        # Get indices for lognormally and reverse lognormally distributed state variables 
        # and observations for this time step
        ln_var_SV,  rl_var_SV  = get_ln_rl_var(ln_vars_SV, rl_vars_SV, n_t_obs,ii+1,x_a_model[:,-1])
        ln_var_obs, rl_var_obs = get_ln_rl_var(ln_vars_obs,rl_vars_obs,n_t_obs,ii+1,y[:,ii+1])

        # Update the background error covariance matrix
        if ii > 0:
            # If previous 3DVAR was unsuccessful, set x_a to nan
            if (not res.success) or timeout:
                x_a[:,ii] = np.nan*np.empty(n_SV)
                timeout = False
            else:
                # Only update when minimization was successful
                B = getBupdate(x_a_model[:,1:],x_b_model[:,1:],ln_var_SV,rl_var_SV,xi_SV)
        Binv = pinv(B)

        # Set bounds
        bnds = np.array([(None, None)]*n_SV)
        bnds[ln_var_SV] = (1e-5, None)          # Lognormal variables must be positive
        bnds[rl_var_SV] = (None, xi_SV-1e-5)    # Reverse lognormal variables must be below xi

        tol = 1e-5
        if method == 'root':
            res = root( \
                grad_3D, \
                x_a_model[:,-1], \
                args = (x_a_model[:,-1],y[:,ii+1],H,Binv,Rinv, \
                    ln_var_SV,ln_var_obs,rl_var_SV,rl_var_obs,xi_SV,xi_obs,l_SV,l_obs), \
                jac = False,
                tol = tol \
            )

            if not res.success:
                print('Root finding of the gradient has failed at t = '+str(t_obs[ii+1])+', minimizing instead.')

        if method == 'min' or (method == 'root' and not res.success):
            res = minimize( \
                cost_3D, \
                x_a_model[:,-1], \
                args = (x_a_model[:,-1],y[:,ii+1],H,Binv,Rinv, \
                    ln_var_SV,ln_var_obs,rl_var_SV,rl_var_obs,xi_SV,xi_obs,l_SV,l_obs), \
                bounds = bnds, \
                jac = True, \
                tol = tol \
            )
            if not res.success:
                print('Minimization of the cost function has failed at t = '+str(t_obs[ii+1]))
                

        # Set analysis state from solution
        x_a[:,ii+1] = res.x

    # Forecasting step
    ii += 1
    tt = ii * n_t_mod       # Index for model time at observation point
    tt_next = tt + n_t_mod  # Index for model time at next observation point
    
    # Run the model for the analysis
    sim_time = t_true[tt:tt_next+1]
    try:
        with time_limit(60):
            x_a_model = model(sim_time, x_a[:,ii])

        # Save model run in background
        x_b[:,tt+1:tt_next+1] = x_a_model[:,1:]
    except TimeoutException:
        # Save model run in background
        x_b[:,tt+1:tt_next+1] = np.nan*np.empty(sim_time.size-1)
        
    return x_a, x_b, t_true


def cost_3D(x, x_b, y, H, Binv, Rinv, \
    ln_var_SV, ln_var_obs, \
    rl_var_SV, rl_var_obs, xi_SV, xi_obs, l_SV, l_obs):
    """
    Cost function for the 3DVAR assimilation step, which needs to be minimized
    to obtain the optimal analysis state from the background and observations
    
    #### Input
    - `x`           ->  State to be optimized, vector of size n_SV
    - `x_b`         ->  Background state, vector of size n_SV
    - `y`           ->  Observations, vector of size n_obs
    - `H`           ->  Observation operator, obtained from create_H, 
                        array of size n_obs x n_SV
    - `Binv`        ->  Inverse of the background error covariance matrix, array of size n_SV x n_SV
    - `Rinv`        ->  Inverse of the observation error covariance matrix, array of size n_obs x n_obs
                        This matrix is assumed to be constant throughout the assimilation
    - `ln_vars_SV`  ->  State variables that should be treated lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `ln_vars_obs` ->  Observation variables that should be treated lognormally, 
                        as an array of indices between 0 and n_obs-1. 
    - `rl_vars_SV`  ->  State variables that should be treated reverse lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `rl_vars_obs` ->  Observation variables that should be treated reverse lognormally, 
                        as an array of indices between 0 and n_obs-1. 
    - `xi_SV`       ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`      ->  Parameter for the reverse lognormal distribution for the observations
    - `l_SV`        ->  Coefficient to use in the cost function of the background
                        to select the descriptor to optimize
                            l_SV  =  1.0    ->  mode
                            l_SV  = -0.5    ->  mean
                            l_SV  =  0.0    ->  median
    - `l_obs`       ->  Coefficient to use in the cost function of the observations
                        to select the descriptor to optimize
                            l_obs =  1.0    ->  mode
                            l_obs = -0.5    ->  mean
                            l_obs =  0.0    ->  median

    #### Output
    - `J`           ->  Cost function for 3DVAR
    - `grad`        ->  Gradient of the cost function
    """

    # Transform background and observations to mixed variables
    Hx = H @ x
    x_mix, x_b_mix, ln_SV, rl_SV\
        = transform_vars(x, x_b, ln_var_SV, rl_var_SV, xi_SV)
    y_mix, Hx_mix, ln_obs, rl_obs \
        = transform_vars(y, Hx, ln_var_obs, rl_var_obs, xi_obs)

    # Background cost
    J_bg = (x_mix - x_b_mix) @ Binv @ (x_mix - x_b_mix)/2 \
        + l_SV*np.sum(x_mix[ln_SV] - x_b_mix[ln_SV]) \
        + l_SV*np.sum(x_mix[rl_SV] - x_b_mix[rl_SV])

    # Observational cost
    J_obs = (y_mix - Hx_mix) @ Rinv @ (y_mix - Hx_mix)/2 \
        + l_obs*np.sum(y_mix[ln_obs] - Hx_mix[ln_obs]) \
        + l_obs*np.sum(y_mix[rl_obs] - Hx_mix[rl_obs])

    # Gradient of the cost function
    gradX = np.ones_like(x)             # dX/dx
    gradX[ln_SV] = 1/x[ln_SV]           # d(log x)/dx
    gradX[rl_SV] = 1/(x[rl_SV]-xi_SV)   # d(log (xi - x))/dx
    
    gradHX = copy.copy(H)
    HxMat = np.tile(Hx,(x.size,1)).T
    gradHX[ln_obs,:] = H[ln_obs,:]/HxMat[ln_obs,:]
    gradHX[rl_obs,:] = H[rl_obs,:]/(HxMat[rl_obs,:]-xi_obs)

    # Background
    gradJ_bg = gradX* (Binv @ (x_mix - x_b_mix))
    gradJ_bg[ln_SV] += l_SV*gradX[ln_SV]
    gradJ_bg[rl_SV] += l_SV*gradX[rl_SV]

    # Observations
    gradJ_obs = -gradHX.T @ Rinv @ (y_mix - Hx_mix)
    gradJ_obs += -l_obs*np.sum(gradHX[ln_obs,:], axis = 0)
    gradJ_obs += -l_obs*np.sum(gradHX[rl_obs,:], axis = 0)

    return J_bg + J_obs, gradJ_bg + gradJ_obs


def grad_3D(x, x_b, y, H, Binv, Rinv, \
    ln_var_SV, ln_var_obs, \
    rl_var_SV, rl_var_obs, xi_SV, xi_obs, l_SV, l_obs):
    """
    Gradient of the cost function for the 3DVAR assimilation step, for which a root needs to be found
    to obtain the optimal analysis state from the background and observations
    
    #### Input
    - `x`           ->  State to be optimized, vector of size n_SV
    - `x_b`         ->  Background state, vector of size n_SV
    - `y`           ->  Observations, vector of size n_obs
    - `H`           ->  Observation operator, obtained from create_H, 
                        array of size n_obs x n_SV
    - `Binv`        ->  Inverse of the background error covariance matrix, array of size n_SV x n_SV
    - `Rinv`        ->  Inverse of the observation error covariance matrix, array of size n_obs x n_obs
                        This matrix is assumed to be constant throughout the assimilation
    - `ln_vars_SV`  ->  State variables that should be treated lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `ln_vars_obs` ->  Observation variables that should be treated lognormally, 
                        as an array of indices between 0 and n_obs-1. 
    - `rl_vars_SV`  ->  State variables that should be treated reverse lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `rl_vars_obs` ->  Observation variables that should be treated reverse lognormally, 
                        as an array of indices between 0 and n_obs-1. 
    - `xi_SV`       ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`      ->  Parameter for the reverse lognormal distribution for the observations
    - `l_SV`        ->  Coefficient to use in the cost function of the background
                        to select the descriptor to optimize
                            l_SV  =  1.0    ->  mode
                            l_SV  = -0.5    ->  mean
                            l_SV  =  0.0    ->  median
    - `l_obs`       ->  Coefficient to use in the cost function of the observations
                        to select the descriptor to optimize
                            l_obs =  1.0    ->  mode
                            l_obs = -0.5    ->  mean
                            l_obs =  0.0    ->  median

    #### Output
    - `grad`        ->  Gradient of the cost function
    """

    # Transform background and observations to mixed variables
    Hx = H @ x
    x_mix, x_b_mix, ln_SV, rl_SV\
        = transform_vars(x, x_b, ln_var_SV, rl_var_SV, xi_SV)
    y_mix, Hx_mix, ln_obs, rl_obs \
        = transform_vars(y, Hx, ln_var_obs, rl_var_obs, xi_obs)

    # Gradient of the cost function
    gradX = np.ones_like(x)             # dX/dx
    gradX[ln_SV] = 1/x[ln_SV]           # d(log x)/dx
    gradX[rl_SV] = 1/(x[rl_SV]-xi_SV)   # d(log (xi - x))/dx
    
    gradHX = copy.copy(H)
    HxMat = np.tile(Hx,(x.size,1)).T
    gradHX[ln_obs,:] = H[ln_obs,:]/HxMat[ln_obs,:]
    gradHX[rl_obs,:] = H[rl_obs,:]/(HxMat[rl_obs,:]-xi_obs)

    # Background
    gradJ_bg = gradX* (Binv @ (x_mix - x_b_mix))
    gradJ_bg[ln_SV] += l_SV*gradX[ln_SV]
    gradJ_bg[rl_SV] += l_SV*gradX[rl_SV]

    # Observations
    gradJ_obs = -gradHX.T @ Rinv @ (y_mix - Hx_mix)
    gradJ_obs += -l_obs*np.sum(gradHX[ln_obs,:], axis = 0)
    gradJ_obs += -l_obs*np.sum(gradHX[rl_obs,:], axis = 0)

    return gradJ_bg + gradJ_obs

def hess_3D(x, x_b, y, H, Binv, Rinv, \
    ln_var_SV, ln_var_obs, \
    rl_var_SV, rl_var_obs, xi_SV, xi_obs, l_SV, l_obs):
    """
    Gradient and Hessian of the cost function for the 3DVAR assimilation step, 
    for which a root needs to be found to obtain the optimal analysis state 
    from the background and observations
    
    #### Input
    - `x`           ->  State to be optimized, vector of size n_SV
    - `x_b`         ->  Background state, vector of size n_SV
    - `y`           ->  Observations, vector of size n_obs
    - `H`           ->  Observation operator, obtained from create_H, 
                        array of size n_obs x n_SV
    - `Binv`        ->  Inverse of the background error covariance matrix, array of size n_SV x n_SV
    - `Rinv`        ->  Inverse of the observation error covariance matrix, array of size n_obs x n_obs
                        This matrix is assumed to be constant throughout the assimilation
    - `ln_vars_SV`  ->  State variables that should be treated lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `ln_vars_obs` ->  Observation variables that should be treated lognormally, 
                        as an array of indices between 0 and n_obs-1. 
    - `rl_vars_SV`  ->  State variables that should be treated reverse lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `rl_vars_obs` ->  Observation variables that should be treated reverse lognormally, 
                        as an array of indices between 0 and n_obs-1. 
    - `xi_SV`       ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`      ->  Parameter for the reverse lognormal distribution for the observations
    - `l_SV`        ->  Coefficient to use in the cost function of the background
                        to select the descriptor to optimize
                            l_SV  =  1.0    ->  mode
                            l_SV  = -0.5    ->  mean
                            l_SV  =  0.0    ->  median
    - `l_obs`       ->  Coefficient to use in the cost function of the observations
                        to select the descriptor to optimize
                            l_obs =  1.0    ->  mode
                            l_obs = -0.5    ->  mean
                            l_obs =  0.0    ->  median

    #### Output
    - `hess`        ->  Hessian of the cost function
    """

    # Transform background and observations to mixed variables
    Hx = H @ x
    x_mix, x_b_mix, ln_SV, rl_SV\
        = transform_vars(x, x_b, ln_var_SV, rl_var_SV, xi_SV)
    y_mix, Hx_mix, ln_obs, rl_obs \
        = transform_vars(y, Hx, ln_var_obs, rl_var_obs, xi_obs)

    # Hessian of the cost function
    laplX = np.zeros_like(x)                # d^2X/dx^2
    laplX[ln_SV] = -1/x[ln_SV]**2           # d^2(log x)/dx^2
    laplX[rl_SV] = -1/(x[rl_SV]-xi_SV)**2   # d^2(log (xi-x))/dx^2

    laplHX = np.zeros((x.size,x.size,y.size))
    for iy in range(y.size):
        if iy in ln_obs:
            laplHX[:,:,iy] = -np.outer(H[iy,:],H[iy,:])/Hx[iy]**2
        elif iy in rl_obs:
            laplHX[:,:,iy] = -np.outer(H[iy,:],H[iy,:])/(Hx[iy]-xi_obs)**2

    # Background
    laplJ_bg = np.diag(laplX* (Binv @ (x_mix - x_b_mix)))
    laplJ_bg += np.diag(gradX) @ Binv @ np.diag(gradX)
    laplJ_bg[ln_SV,ln_SV] += l_SV*laplX[ln_SV]
    laplJ_bg[rl_SV,rl_SV] += l_SV*laplX[rl_SV]

    # Observations
    laplJ_obs = -np.dot(laplHX, Rinv @ (y_mix - Hx_mix))
    laplJ_obs += gradHX.T @ Rinv @ gradHX
    laplJ_obs += -l_obs*np.sum(laplHX[:,:,ln_obs], axis = -1)
    laplJ_obs += -l_obs*np.sum(laplHX[:,:,rl_obs], axis = -1)


    return laplJ_bg + laplJ_obs


####################################################################################
####################################################################################
####################################################################################




def var4d(init_guess, t_obs, t_DA, n_t_mod, n_t_mod_obs, y, H, B, R, model, model_tlm, \
    ln_vars_SV = [], ln_vars_obs = [], \
    rl_vars_SV = [], rl_vars_obs = [], xi_SV = 0.0, xi_obs = 0.0, method = 'root'):
    """
    4DVAR

    4DVAR data assimilation technique for use in a general model, which allows for 
    Gaussian and mixed Gaussian-lognormal observations and background state variables

    #### Input
    - `init_guess`  ->  Initial guess for the model analysis, vector of size n_SV
    - `t_obs`       ->  Time values of observations, vector of size n_t_obs
    - `t_DA`        ->  Time values of analysis window, vector of size n_win_DA
    - `n_t_mod`     ->  Number of time steps to use in each model run between analysis times
    - `n_t_mod_obs` ->  Number of time steps to use in each model run between observation times
    - `y`           ->  Observations, array of size n_obs x n_t_obs
    - `H`           ->  Observation operator, obtained from create_H, 
                        array of size n_obs x n_SV
    - `B`           ->  Background error covariance matrix, array of size n_SV x n_SV
    - `R`           ->  Observation error covariance matrix, array of size n_obs x n_obs
                        This matrix is assumed to be constant throughout the assimilation
    - `model`       ->  Model to use in the analysis, function of the form
                            x = model(t,x_0),
                        where t contains the time values where the model is evaluated
                        and x_0 is the initial condition;
                        the output x are the state variables for all time values
    - `model_tlm`   ->  Tangent linear model to use in the analysis, function of the form
                            x, M = model_tlm(t,x_0),
                        where t contains the time values where the model is evaluated
                        and x_0 is the initial condition;
                        the output x are the state variables for all time values,
                        and the tangent linear model matrix M
    - `ln_vars_SV`  ->  State variables that should be treated lognormally, 
                        * If ln_vars is a callable, it should be of the form
                                ln_var = ln_vars(SV_input),
                          where SV_input contains the state variables at one time step
                          and ln_var is a list of indices between 0 and n_SV-1
                        * If ln_vars is a list of indices between 0 and n_SV-1,
                          the state variables of these indices are treated as 
                          lognormally distributed for all time steps
                        * If ln_vars is a list of lists, the length of the top list
                          should be n_t_obs, and the inner lists are indices between 0 and n_SV-1,
                          the state variables of these indices are treated as 
                          lognormally distributed for each specific time step
                        Default is [] for all Gaussian variables
    - `ln_vars_obs` ->  Observations that should be treated lognormally, 
                        * If ln_vars is a callable, it should be of the form
                                ln_var = ln_vars(y_input),
                          where y_input contains the observed variables at one time step
                          and ln_var is a list of indices between 0 and n_obs-1
                        * If ln_vars is a list of indices between 0 and n_obs-1,
                          the observations of these indices are treated as 
                          lognormally distributed for all time steps
                        * If ln_vars is a list of lists, the length of the top list
                          should be n_t_obs, and the inner lists are indices between 0 and n_obs-1,
                          the observations of these indices are treated as 
                          lognormally distributed for each specific time step
                        Default is [] for all Gaussian variables
    - `rl_vars_SV`  ->  State variables that should be treated reverse lognormally, same as ln_vars_SV
    - `rl_vars_obs` ->  Observations that should be treated reverse lognormally, same as ln_vars_obs
    - `xi_SV`       ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`      ->  Parameter for the reverse lognormal distribution for the observations
    - `method`      ->  Method for analysis, either `root` (default) for finding the root of the gradient,
                        or `min` for finding the minimum of the cost function

    #### Output
    - `x_b`         ->  Background state, array of size n_SV x n_t
    - `x_a`         ->  Analysis state, array of size n_SV x n_t_obs
    - `t_true`      ->  Total time for background, vector of size n_t
    """
    
    # Save lengths of arrays
    n_SV = init_guess.size       # number of state variables
    n_t_obs = t_obs.size         # number of observations
    n_win_DA = t_DA.size         # number of DA windows
    n_t = n_win_DA * n_t_mod + 1 # number of total time steps

    # Initialize background and analysis states
    x_b = np.empty((n_SV,n_t))
    x_a = np.empty((n_SV,n_win_DA))

    # At t=0, background state is set to the initial guess
    x_b[:,0] = init_guess
    x_e = init_guess
    timeout = False

    # Total time evolution for model
    dt_DA = t_DA[1] - t_DA[0] # Assuming evenly spaced DA windows
    t_true = np.linspace(t_DA[0],t_DA[-1] + dt_DA, n_t) # Assuming one prediction window

    # Loop over all DA windows
    for ii in range(n_win_DA):
        # Calculated states are on model time, and not DA time
        tt = ii * n_t_mod                # Index for model time at DA window
        tt_next = tt + n_t_mod           # Index for model time at next DA window
        sim_time = t_true[tt:tt_next+1]  # Simulation time of DA window

        # Find all observations within DA window
        # ind_obs = np.where(sim_time[0] < t_obs <= sim_time[-1])
        ind_obs = np.nonzero(np.logical_and(sim_time[0] < t_obs, t_obs <= sim_time[-1]))[0]
        n_obs_win = len(ind_obs)
        if n_obs_win == 0:
            raise RuntimeError("No observations found in assimilation window!")
        y_win = y[:,ind_obs]
        t_obs_win = np.insert(t_obs[ind_obs], 0, sim_time[0])

        # Get lognormally distributed state variables for all observation times in this window
        if callable(ln_vars_SV):
            ln_var_SV = ln_vars_SV(x_b[:,tt])
        elif len(ln_vars_SV) == n_t_obs:
            ln_var_SV = ln_vars_SV[ii]
        else: # Same treatment for all time steps
            ln_var_SV = ln_vars_SV
        # Note: ln_var_SV is a list of indices between 0 and n_SV-1

        # Get lognormally distributed observations for all observation times in this window
        if callable(ln_vars_obs):
            ln_var_obs = ln_vars_obs(y_win)
        elif len(ln_vars_obs) == n_t_obs:
            ln_var_obs = [ln_vars_obs[i_obs] for i_obs in ind_obs]
        else: # Same treatment for all time steps
            ln_var_obs = [ln_vars_obs] * n_obs_win
        # Note: ln_var_obs is a nested list of length n_obs_win (# of observations in the DA window),
        #       its elements are lists of indices between 0 and n_obs-1

        # Get reverse lognormally distributed state variables for all observation times in this window
        if callable(rl_vars_SV):
            rl_var_SV = rl_vars_SV(x_b[:,tt])
        elif len(rl_vars_SV) == n_t_obs:
            rl_var_SV = rl_vars_SV[ii]
        else: # Same treatment for all time steps
            rl_var_SV = rl_vars_SV

        # Get reverse lognormally distributed observations for all observation times in this window
        if callable(rl_vars_obs):
            rl_var_obs = rl_vars_obs(y_win)
        elif len(rl_vars_obs) == n_t_obs:
            rl_var_obs = [rl_vars_obs[i_obs] for i_obs in ind_obs]
        else: # Same treatment for all time steps
            rl_var_obs = [rl_vars_obs] * n_obs_win

        # Check if there is any overlap in lognormal and reverse lognormal variables
        s_ln_var_SV = set(ln_var_SV)
        if any(x in s_ln_var_SV for x in rl_var_SV):
            raise RuntimeError("State variables can not be lognormal and reverse lognormal at the same time!")
        for j_obs in range(n_obs_win):
            s_ln_var_obs = set(ln_var_obs[j_obs])
            if any(x in s_ln_var_obs for x in rl_var_obs[j_obs]):
                raise RuntimeError("Observations can not be lognormal and reverse lognormal at the same time!")


        # Update B matrix
        if ii > 0:
            if (not res.success) or timeout:
                print('Assimilation failed at t = '+str(t_DA[ii-1]))
                x_a[:,ii-1] = np.nan*np.empty(n_SV)
                timeout = False
            else:
                B = getBupdate(x_a_model[:,1:],x_b_model[:,1:],ln_var_SV,rl_var_SV,xi_SV)
                #!!! Should the B update include a time-dependent distribution as well?

        # Invert error covariance matrices
        Binv = pinv(B)
        Rinv = pinv(R)

        # Set bounds
        bnds = [(None, None)]*n_SV
        bnds = np.array(bnds)
        bnds[ln_var_SV] = (1e-5, None)
        bnds[rl_var_SV] = (None, xi_SV-1e-5)

        # Minimize cost function
        tol = 1e-5
        if method == 'root':
            res = root( \
                grad_4D, \
                x_b[:,tt], \
                args = (x_b[:,tt], y_win, t_obs_win, n_t_mod_obs, H, Binv, Rinv, model_tlm, \
                        ln_var_SV, ln_var_obs, rl_var_SV, rl_var_obs, xi_SV, xi_obs), \
                jac = False,
                tol = tol \
            )
            if res.nfev > 6:
                print("Number of iterations = "+str(res.nfev)+' at t = '+str(t_DA[ii]))
        if method == 'min':
            res = minimize( \
                cost_4D, \
                x_b[:,tt], \
                args = (x_b[:,tt], y_win, t_obs_win, n_t_mod_obs, H, Binv, Rinv, model_tlm, \
                        ln_var_SV, ln_var_obs, rl_var_SV, rl_var_obs, xi_SV, xi_obs), \
                jac = True, \
                bounds = bnds, \
                tol = tol \
            )
            if res.nit > 4:
                print("Number of iterations = "+str(res.nit)+' at t = '+str(t_DA[ii]))

        # Set analysis state from solution
        x_a[:,ii] = res.x

        try:
            with time_limit(60):
                x_a_model = model(sim_time, res.x)     # Model run for analysis state
                x_b_model = model(sim_time, x_e)       # Model run for background state (needed for B update)

            # Save model run in background
            x_b[:,tt+1:tt_next+1] = x_a_model[:,1:]
            # x_e = x_a_model[:,-1]
            x_e = x_b_model[:,-1]

        except TimeoutException:
            print("Warning: timeout in model run, skipping this assimilation step!")
            print("    x_a_init = "+str(res.x))
            print("    x_b_init = "+str(x_b[:,tt]))

            # If timeout, something went wrong in analysis
            x_a[:,ii] = np.nan*np.empty(n_SV)
            x_b[:,tt:tt_next+1] = np.nan*np.empty_like(sim_time)

            # Restart DA from initial guess next step
            x_b[:,tt_next] = init_guess
            x_e = init_guess
            timeout = True
        
    return x_a, x_b, t_true

def cost_4D(x, x_b, y, t_obs, n_t_mod_obs, H, Binv, Rinv, model, \
    ln_var_SV, ln_var_obs, \
    rl_var_SV, rl_var_obs, xi_SV, xi_obs):
    """
    Cost function for the 4DVAR assimilation step, which needs to be minimized
    to obtain the optimal analysis state from the background and observations
    
    #### Input
    - `x`           ->  State to be optimized, vector of size n_SV
    - `x_b`         ->  Background state, vector of size n_SV
    - `y`           ->  Observations, vector of size n_obs x n_obs_win
    - `t_obs`       ->  Observation times, vector of size n_obs_win + 1 (first value is analysis time)
    - `H`           ->  Observation operator, obtained from create_H, 
                        array of size n_obs x n_SV
    - `Binv`        ->  Inverse of the background error covariance matrix, array of size n_SV x n_SV
    - `Rinv`        ->  Inverse of the observation error covariance matrix, array of size n_obs x n_obs
                        This matrix is assumed to be constant throughout the assimilation
    - `model`       ->  Model to use in the analysis, function of the form
                            x, M = model_tlm(t,x_0),
                        where t contains the time values where the model is evaluated
                        and x_0 is the initial condition;
                        the output x are the state variables for all time values,
                        and the tangent linear model matrix M for use in the Jacobian
    - `ln_vars_SV`  ->  State variables that should be treated lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `ln_vars_obs` ->  Observation variables that should be treated lognormally, 
                        a list of length n_obs_win containing lists of indices between 0 and n_obs-1.
    - `rl_vars_SV`  ->  State variables that should be treated reverse lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `rl_vars_obs` ->  Observation variables that should be treated reverse lognormally, 
                        a list of length n_obs_win containing lists of indices between 0 and n_obs-1. 
    - `xi_SV`       ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`      ->  Parameter for the reverse lognormal distribution for the observations

    #### Output
    - `J`           ->  Cost function for 4DVAR
    - `grad`        ->  Gradient of the cost function
    """

    # Model run for observations
    n_obs_win = t_obs.size - 1
    t_obs_long = np.empty(n_obs_win * n_t_mod_obs + 1)
    for ii in range(n_obs_win):
        tt = n_t_mod_obs*ii
        t_obs_long[tt:(tt+n_t_mod_obs+1)] = np.linspace(t_obs[ii],t_obs[ii+1],n_t_mod_obs+1)
    try:
        with time_limit(60):
            x_obs_long, M_long = model(t_obs_long, x)
    except TimeoutException:
        print("Warning: timeout in model run for x="+str(x)+", trying with background.")
        with time_limit(60):
            # If this also fails, raise error
            x_obs_long, M_long = model(t_obs_long, x_b)
    x_obs = x_obs_long[:,::n_t_mod_obs]
    x_obs = x_obs[:,1:]
    M = M_long[:,:,::n_t_mod_obs]
    M = M[:,:,1:]
    Hx = H @ x_obs

    # Transform background to mixed variables
    x_mix, x_b_mix, ln_SV, rl_SV\
        = transform_vars(x, x_b, ln_var_SV, rl_var_SV, xi_SV)

    # Transform observations to mixed variables
    y_mix = copy.copy(y)
    Hx_mix = copy.copy(Hx)
    ln_obs = copy.copy(ln_var_obs)
    rl_obs = copy.copy(rl_var_obs)
    for i_obs in range(n_obs_win):
        y_mix[:,i_obs], Hx_mix[:,i_obs], ln_obs[i_obs], rl_obs[i_obs] \
            = transform_vars(y[:,i_obs], Hx[:,i_obs], ln_var_obs[i_obs], rl_var_obs[i_obs], xi_obs)

    # Background cost
    J_bg = (x_mix - x_b_mix) @ Binv @ (x_mix - x_b_mix)/2 \
        + np.sum(x_mix[ln_SV] - x_b_mix[ln_SV]) \
        + np.sum(x_mix[rl_SV] - x_b_mix[rl_SV])

    # Observational cost
    J_obs = 0.0
    for i_obs in range(n_obs_win):
        J_obs += (y_mix[:,i_obs] - Hx_mix[:,i_obs]) @ Rinv @ (y_mix[:,i_obs] - Hx_mix[:,i_obs])/2 \
            + np.sum(y_mix[ln_obs[i_obs],i_obs] - Hx_mix[ln_obs[i_obs],i_obs]) \
            + np.sum(y_mix[rl_obs[i_obs],i_obs] - Hx_mix[rl_obs[i_obs],i_obs])

    # Gradient of the cost function
    gradX = np.ones_like(x)                     # dX/dx
    gradX[ln_SV] = 1/x[ln_SV]           # d(log x)/dx
    gradX[rl_SV] = 1/(x[rl_SV]-xi_SV)   # d(log (xi - x))/dx

    # Background
    gradJ_bg = gradX* (Binv @ (x_mix - x_b_mix))
    gradJ_bg[ln_SV] += gradX[ln_SV]
    gradJ_bg[rl_SV] += gradX[rl_SV]
    
    gradJ_obs = np.zeros_like(x)
    for i_obs in range(n_obs_win):
        gradHX = copy.copy(H)
        HxMat = np.tile(Hx[:,i_obs],(x.size,1)).T
        gradHX[ln_obs[i_obs],:] = H[ln_obs[i_obs],:]/HxMat[ln_obs[i_obs],:]
        gradHX[rl_obs[i_obs],:] = H[rl_obs[i_obs],:]/(HxMat[rl_obs[i_obs],:]-xi_obs)

        # Tangent linear model
        gradHX = gradHX @ M[:,:,i_obs]

        # Observations
        gradJ_obs += -gradHX.T @ Rinv @ (y_mix[:,i_obs] - Hx_mix[:,i_obs])
        gradJ_obs += -np.sum(gradHX[ln_obs[i_obs],:], axis = 0)
        gradJ_obs += -np.sum(gradHX[rl_obs[i_obs],:], axis = 0)

    return J_bg + J_obs, gradJ_bg + gradJ_obs


def grad_4D(x, x_b, y, t_obs, n_t_mod_obs, H, Binv, Rinv, model, \
    ln_var_SV, ln_var_obs, \
    rl_var_SV, rl_var_obs, xi_SV, xi_obs):
    """
    Gradient of the cost function for the 4DVAR assimilation step, 
    for which a root needs to be found to obtain the optimal analysis state 
    from the background and observations
    
    #### Input
    - `x`           ->  State to be optimized, vector of size n_SV
    - `x_b`         ->  Background state, vector of size n_SV
    - `y`           ->  Observations, vector of size n_obs x n_obs_win
    - `t_obs`       ->  Observation times, vector of size n_obs_win + 1 (first value is analysis time)
    - `H`           ->  Observation operator, obtained from create_H, 
                        array of size n_obs x n_SV
    - `Binv`        ->  Inverse of the background error covariance matrix, array of size n_SV x n_SV
    - `Rinv`        ->  Inverse of the observation error covariance matrix, array of size n_obs x n_obs
                        This matrix is assumed to be constant throughout the assimilation
    - `model`       ->  Model to use in the analysis, function of the form
                            x, M = model_tlm(t,x_0),
                        where t contains the time values where the model is evaluated
                        and x_0 is the initial condition;
                        the output x are the state variables for all time values,
                        and the tangent linear model matrix M for use in the Jacobian
    - `ln_vars_SV`  ->  State variables that should be treated lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `ln_vars_obs` ->  Observation variables that should be treated lognormally, 
                        a list of length n_obs_win containing lists of indices between 0 and n_obs-1.
    - `rl_vars_SV`  ->  State variables that should be treated reverse lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `rl_vars_obs` ->  Observation variables that should be treated reverse lognormally, 
                        a list of length n_obs_win containing lists of indices between 0 and n_obs-1. 
    - `xi_SV`       ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`      ->  Parameter for the reverse lognormal distribution for the observations

    #### Output
    - `grad`        ->  Gradient of the cost function
    """

    # Model run for observations
    n_obs_win = t_obs.size - 1
    t_obs_long = np.empty(n_obs_win * n_t_mod_obs + 1)
    for ii in range(n_obs_win):
        tt = n_t_mod_obs*ii
        t_obs_long[tt:(tt+n_t_mod_obs+1)] = np.linspace(t_obs[ii],t_obs[ii+1],n_t_mod_obs+1)
    try:
        with time_limit(60):
            x_obs_long, M_long = model(t_obs_long, x)
    except TimeoutException:
        print("Warning: timeout in model run for x="+str(x)+", trying with background.")
        with time_limit(60):
            x_obs_long, M_long = model(t_obs_long, x_b)
    x_obs = x_obs_long[:,::n_t_mod_obs]
    x_obs = x_obs[:,1:]
    M = M_long[:,:,::n_t_mod_obs]
    M = M[:,:,1:]
    Hx = H @ x_obs
    Hx_mix = copy.copy(Hx)
    

    # Transform background to mixed variables
    x_mix, x_b_mix, ln_SV, rl_SV\
        = transform_vars(x, x_b, ln_var_SV, rl_var_SV, xi_SV)

    # Transform observations to mixed variables
    y_mix = copy.copy(y)
    Hx_mix = copy.copy(Hx)
    ln_obs = copy.copy(ln_var_obs)
    rl_obs = copy.copy(rl_var_obs)
    for i_obs in range(n_obs_win):
        y_mix[:,i_obs], Hx_mix[:,i_obs], ln_obs[i_obs], rl_obs[i_obs] \
            = transform_vars(y[:,i_obs], Hx[:,i_obs], ln_var_obs[i_obs], rl_var_obs[i_obs], xi_obs)

    # Gradient of the cost function
    gradX = np.ones_like(x)                     # dX/dx
    gradX[ln_SV] = 1/x[ln_SV]           # d(log x)/dx
    gradX[rl_SV] = 1/(x[rl_SV]-xi_SV)   # d(log (xi - x))/dx

    # Background
    gradJ_bg = gradX* (Binv @ (x_mix - x_b_mix))
    gradJ_bg[ln_SV] += gradX[ln_SV]
    gradJ_bg[rl_SV] += gradX[rl_SV]
    
    gradJ_obs = np.zeros_like(x)
    for i_obs in range(n_obs_win):
        gradHX = copy.copy(H)
        HxMat = np.tile(Hx[:,i_obs],(x.size,1)).T
        gradHX[ln_obs[i_obs],:] = H[ln_obs[i_obs],:]/HxMat[ln_obs[i_obs],:]
        gradHX[rl_obs[i_obs],:] = H[rl_obs[i_obs],:]/(HxMat[rl_obs[i_obs],:]-xi_obs)

        # Tangent linear model
        gradHX = gradHX @ M[:,:,i_obs]

        # Observations
        gradJ_obs += -gradHX.T @ Rinv @ (y_mix[:,i_obs] - Hx_mix[:,i_obs])
        gradJ_obs += -np.sum(gradHX[ln_obs[i_obs],:], axis = 0)
        gradJ_obs += -np.sum(gradHX[rl_obs[i_obs],:], axis = 0)

    return gradJ_bg + gradJ_obs
