""" 
Functions needed for the maximum likelihood ensemble filter. 

#### Methods:
- `MLEF`            ->  Maximum likelihood ensemble filter
- `MLES`            ->  Maximum likelihood ensemble smoother

#### Author:
Senne Van Loon
Cooperative Institute for Research in the Atmosphere (CIRA),
Colorado State University,
3925A West Laporte Ave, Fort Collins, CO 80521

#### References and acknowledgements:
[1] Fletcher, S. J. (2017). Data assimilation for the geosciences: From theory to application. Elsevier
[2] Steven J. Fletcher, Milija Zupanski, Michael R. Goodliff, Anton J. Kliewer, Andrew S. Jones, John M. Forsythe, Ting-Chi Wu, Md. Jakir Hossen, and Senne van Loon. Lognormal and Mixed Gaussian-Lognormal Kalman Filters
[3] Zupanski, M., 2005. Maximum likelihood ensemble filter: Theoretical aspects. Monthly Weather Review, 133(6), pp.1710-1726.
[4] Steven J. Fletcher, Milija Zupanski, Michael R. Goodliff, Anton J. Kliewer, Andrew S. Jones, John M. Forsythe, Ting-Chi Wu. Maximum Likelihood Ensemble Smoother

"""

# Load modules
import numpy as np
from scipy.linalg import inv, cholesky
from scipy.optimize import minimize, root
from mod_DA_general import get_ln_rl_var, transform_vars
import copy

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)



def MLEF(init_guess, t_obs, n_t_mod, n_e, y, h, sqrtP_a_mix, R, model, \
        ln_vars_SV = [], ln_vars_obs = [], \
        rl_vars_SV = [], rl_vars_obs = [], xi_SV = 0.0, xi_obs = 0.0):
    """
    Maximum likelihood ensemble filter

    Maximum likelihood ensemble filter data assimilation technique for use in a general model, 
    which allows for mixed Gaussian, lognormal, and reverse lognormal observations 
    and background state variables. This version uses Hessian preconditioning to minimize the cost function

    #### Input
    - `init_guess`  ->  Initial guess for the model analysis, vector of size n_SV
    - `t_obs`       ->  Time values of observations, vector of size n_t_obs
    - `n_t_mod`     ->  Number of time steps to use in each model run
    - `n_e`         ->  Number of ensemble members
    - `y`           ->  Observations, array of size n_obs x n_t_obs
    - `h`           ->  Observation operator, a function of the form
                            y = h(x)
    - `sqrtP_a_mix` ->  Square root of the analysis error covariance matrix (initial), 
                        array of size n_SV x n_e
    - `R`           ->  Observation error covariance matrix, array of size n_obs x n_obs
                        This matrix is assumed to be constant throughout the assimilation
    - `model`       ->  Model to use in the analysis, function of the form
                            x = model(t,x_0),
                        where t contains the time values where the model is evaluated
                        and x_0 is the initial condition;
                        the output x are the state variables for all time values
    - `ln_vars_SV`  ->  State variables that should be treated lognormally, 
                        *   If ln_vars is a callable, it should be of the form
                                    ln_var = ln_vars(SV_input),
                            where SV_input contains the state variables at one time step
                            and ln_var is a list of indices between 0 and n_SV-1
                        *   If ln_vars is a list of indices between 0 and n_SV-1,
                            the state variables of these indices are treated as 
                            lognormally distributed for all time steps
                        *   If ln_vars is a list of lists, the length of the top list
                            should be n_t_obs, and the inner lists are indices between 0 and n_SV-1,
                            the state variables of these indices are treated as 
                            lognormally distributed for each specific time step
                        Default is [] for all Gaussian variables
    - `ln_vars_obs` ->  Observations that should be treated lognormally, 
                        *   If ln_vars is a callable, it should be of the form
                                    ln_var = ln_vars(y_input),
                            where y_input contains the observed variables at one time step
                            and ln_var is a list of indices between 0 and n_obs-1
                        *   If ln_vars is a list of indices between 0 and n_obs-1,
                            the observations of these indices are treated as 
                            lognormally distributed for all time steps
                        *   If ln_vars is a list of lists, the length of the top list
                            should be n_t_obs, and the inner lists are indices between 0 and n_obs-1,
                            the observations of these indices are treated as 
                            lognormally distributed for each specific time step
                        Default is [] for all Gaussian variables
    - `rl_vars_SV`  ->  State variables that should be treated reverse lognormally, same as ln_vars_SV
    - `rl_vars_obs` ->  Observations that should be treated reverse lognormally, same as ln_vars_obs
    - `xi_SV`       ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`      ->  Parameter for the reverse lognormal distribution for the observations

    #### Output
    - `x_b`         ->  Background state, array of size n_SV x n_t
    - `x_a`         ->  Analysis state, array of size n_SV x n_t_obs
    - `t_true`      ->  Total time for background, vector of size n_t
    """

    # Save lengths of arrays
    n_SV = init_guess.size      # number of state variables
    n_ob, n_t_obs = y.shape     # number of observed variables and of observations
    n_t = n_t_obs * n_t_mod + 1 # number of total time steps

    # Initialize background and analysis states
    x_b = np.empty((n_SV,n_t))
    x_a = np.empty((n_SV,n_t_obs))

    # At t=0, background and analysis states are set to the initial guess
    x_b[:,0] = init_guess
    x_a[:,0] = init_guess

    # Initialize forecast run variables
    ln_var_SV,  rl_var_SV  = get_ln_rl_var(ln_vars_SV, rl_vars_SV, n_t_obs,0,x_a[:,0])
    sqrtP_a = copy.copy(sqrtP_a_mix)
    sqrtP_a[ln_var_SV,:] = np.exp(sqrtP_a_mix[ln_var_SV,:])
    sqrtP_a[rl_var_SV,:] = xi_SV - np.exp(sqrtP_a_mix[rl_var_SV,:])
    x_a_mat = np.broadcast_to(x_a[:,0],(n_e,n_SV)).T # array of size n_SV x n_e
    x_p = x_a_mat + sqrtP_a
    x_p[ln_var_SV,:] = x_a_mat[ln_var_SV,:] * sqrtP_a[ln_var_SV,:]
    x_p[rl_var_SV,:] = xi_SV - (xi_SV - x_a_mat[rl_var_SV,:]) * (xi_SV - sqrtP_a[rl_var_SV,:])

    # Total time evolution for model
    dt_obs = t_obs[1] - t_obs[0] # Assuming evenly spaced observations
    t_true = np.linspace(t_obs[0],t_obs[-1] + dt_obs, n_t) # Assuming one prediction window

    # Inverse of observation error covariance matrix (constant throughout)
    sqrtRinv = cholesky(inv(R))

    # Loop over all observations
    for ii in range(1,n_t_obs):
        # Calculated states are on model time, and not observation time
        tt = ii * n_t_mod       # Index for model time at observation point
        tt_prev = tt - n_t_mod  # Index for model time at previous observation point
        sim_time = t_true[tt_prev:tt+1]

        ##################################################################
        ###################       Forecast step       ####################
        ##################################################################

        # Model run for analysis state and square root errors (array of size n_e x n_SV x n_t)
        x_f = model(sim_time, x_a[:,ii-1])
        b_f = np.array([ model(sim_time, x_p[:,jE]) for jE in range(n_e) ])
        b_f = b_f[:,:,-1].T

        # Save background forecast
        x_b[:,tt_prev+1:tt+1] = x_f[:,1:]
        x_f = x_f[:,-1]

        # Get indices for lognormally and reverse lognormally distributed state variables 
        # and observations for this time step
        ln_var_SV,  rl_var_SV  = get_ln_rl_var(ln_vars_SV, rl_vars_SV, n_t_obs,ii,x_f)
        ln_var_obs, rl_var_obs = get_ln_rl_var(ln_vars_obs,rl_vars_obs,n_t_obs,ii,y[:,ii])

        # Transform forecast to mixed variables
        x_f_mix, b_f_mix, ln_var_SV, rl_var_SV\
            = transform_vars(x_f, b_f, ln_var_SV, rl_var_SV, xi_SV, ensembles = True)

        # Square root forecast error covariance matrix
        sqrtP_f_mix = b_f_mix - np.broadcast_to(x_f_mix,(n_e,n_SV)).T
        
        # Set x_a to nan if previous minimization was unsuccessful
        if ii > 1 and not res.success:
            x_a[:,ii-1] = np.nan*np.empty(n_SV)
        
        ##################################################################
        ###################       Analysis step       ####################
        ##################################################################

        # Calculate C, (I + C)^(-T/2), and (I + C)^(-1) for preconditioning
        sqrtC_noR, sqrtC, invsqrtIpC_t, invIpC = calc_C_3D( \
            x_f, b_f, h, sqrtRinv, n_ob, n_e, \
            ln_var_obs, rl_var_obs, xi_obs)

        # Minimization of the cost function 
        tol = 1e-5
        res = minimize( \
            cost_mlef, \
            np.zeros(n_e), \
            args = (x_f_mix,y[:,ii],h,sqrtP_f_mix, \
                invIpC,invsqrtIpC_t,sqrtC,sqrtRinv, \
                ln_var_SV, ln_var_obs, \
                rl_var_SV, rl_var_obs, xi_SV, xi_obs, \
                # sqrtC_noR \
                None \
            ), \
            method = 'Newton-CG', \
            jac = True, \
            hess = lambda *args : np.eye(n_e), \
            tol = tol \
        )
        if res.nit > 4:
            print("Number of iterations = "+str(res.nit)+' at t = '+str(t_obs[ii]))
        if not res.success:
            # print('Minimization of the cost function has failed at t = '+str(t_obs[ii]))
            print("t = "+str(t_obs[ii])+": "+res.message)
        
        # Calculate and save analysis state
        x_a_mix = x_f_mix + sqrtP_f_mix @ invsqrtIpC_t @ res.x

        # Transform back to normal variables
        x_a[:,ii] = x_a_mix
        x_a[ln_var_SV,ii] = np.exp(x_a_mix[ln_var_SV])
        x_a[rl_var_SV,ii] = xi_SV - np.exp(x_a_mix[rl_var_SV])

        # Update square root analysis error covariance matrix
        b_a_mix = np.broadcast_to(x_a_mix,(n_e,n_SV)).T + sqrtP_f_mix
        b_a = b_a_mix
        b_a[ln_var_SV,:] = np.exp(b_a_mix[ln_var_SV,:])
        b_a[rl_var_SV,:] = xi_SV - np.exp(b_a_mix[rl_var_SV,:])

        _, _, invsqrtIpC_t, _ = calc_C_3D( \
            x_a[:,ii], b_a, h, sqrtRinv, n_ob, n_e, \
            ln_var_obs, rl_var_obs, xi_obs)
        sqrtP_a_mix = sqrtP_f_mix @ invsqrtIpC_t

        # Transform mixed errors for this time step
        sqrtP_a = copy.copy(sqrtP_a_mix)
        sqrtP_a[ln_var_SV,:] = np.exp(sqrtP_a_mix[ln_var_SV,:])
        sqrtP_a[rl_var_SV,:] = xi_SV - np.exp(sqrtP_a_mix[rl_var_SV,:])

        # State variables for forecast error run in next step
        x_a_mat = np.broadcast_to(x_a[:,ii],(n_e,n_SV)).T # array of size n_SV x n_e
        x_p = x_a_mat + sqrtP_a
        x_p[ln_var_SV,:] = x_a_mat[ln_var_SV,:] * sqrtP_a[ln_var_SV,:]
        x_p[rl_var_SV,:] = xi_SV - (xi_SV - x_a_mat[rl_var_SV,:]) * (xi_SV - sqrtP_a[rl_var_SV,:])

    # Final forecasting step
    ii += 1
    tt = ii * n_t_mod       # Index for model time at observation point
    tt_prev = tt - n_t_mod  # Index for model time at previous observation point
    sim_time = t_true[tt_prev:tt+1]

    # Model run
    x_f = model(sim_time, x_a[:,ii-1])
    x_b[:,tt_prev+1:tt+1] = x_f[:,1:]
    
    return x_a, x_b, t_true


def cost_mlef(z, x_b_mix, y, h, sqrtP_f_mix, \
        invIpC, invsqrtIpC_t, sqrtC, sqrtRinv, \
        ln_var_SV, ln_var_obs, \
        rl_var_SV, rl_var_obs, xi_SV, xi_obs, sqrtC_noR = None):
    """
    Cost function for the MLEF assimilation step, which needs to be minimized
    to obtain the optimal analysis state from the background and observations

    #### Input
    - `z`             ->  Preconditioned state to be optimized, vector of size n_e
    - `x_b`           ->  Background state, vector of size n_SV
    - `y`             ->  Observations, vector of size n_obs
    - `h`             ->  Observation operator, a function of the form
                            y = h(x)
    - `sqrtP_f`       ->  Square root of the forecast error covariance matrix, 
                          array of size n_SV x n_e
    - `invIpC`        ->  (I + C)^(-1), array of size n_e x n_e
    - `invsqrtIpC_t`  ->  (I + C)^(-T/2), array of size n_e x n_e
    - `sqrtC`         ->  z = C^(1/2), array of size n_obs x n_e
    - `sqrtC_noR`     ->  H(x * b) - H(x), array of size n_obs x n_e
    - `sqrtRinv`      ->  Inverse square root of the observation error covariance matrix, 
                            array of size n_obs x n_obs
    - `ln_var_SV`     ->  State variables that should be treated lognormally, 
                          as an array of indices between 0 and n_SV-1. 
    - `ln_var_obs`    ->  Observation variables that should be treated lognormally, 
                          as an array of indices between 0 and n_obs-1. 
    - `rl_var_SV`     ->  State variables that should be treated reverse lognormally, 
                          as an array of indices between 0 and n_SV-1. 
    - `rl_var_obs`    ->  Observation variables that should be treated reverse lognormally, 
                          as an array of indices between 0 and n_obs-1. 
    - `xi_SV`         ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`        ->  Parameter for the reverse lognormal distribution for the observations

    #### Output
    - `J`             ->  Cost function for the MLEF
    - `grad`          ->  Gradient of the cost function
    """

    # Convert x_a from z for observations
    x_a_mix = x_b_mix + sqrtP_f_mix @ invsqrtIpC_t @ z
    x_a = copy.copy(x_a_mix)
    x_a[ln_var_SV] = np.exp(x_a_mix[ln_var_SV])
    x_a[rl_var_SV] = xi_SV - np.exp(x_a_mix[rl_var_SV])

    # Observation operator
    Hx = h(x_a)
    y_mix, Hx_mix, ln_var_obs, rl_var_obs \
        = transform_vars(y, Hx, ln_var_obs, rl_var_obs, xi_obs)

    # Cost function
    J = z @ invIpC @ z/2 \
        + (y_mix - Hx_mix).T @ sqrtRinv.T @ sqrtRinv @ (y_mix - Hx_mix)/2.0
    J += np.sum(x_a_mix[ln_var_SV] - x_b_mix[ln_var_SV])
    J += np.sum(x_a_mix[rl_var_SV] - x_b_mix[rl_var_SV])
    J += np.sum(y_mix[ln_var_obs] - Hx_mix[ln_var_obs])
    J += np.sum(y_mix[rl_var_obs] - Hx_mix[rl_var_obs])

    if sqrtC_noR is None:
        n_SV, n_e = sqrtP_f_mix.shape
        n_ob = y.size
        b_a_mix = np.broadcast_to(x_a_mix,(n_e,n_SV)).T + sqrtP_f_mix
        b_a = b_a_mix
        b_a[ln_var_SV,:] = np.exp(b_a_mix[ln_var_SV,:])
        b_a[rl_var_SV,:] = xi_SV - np.exp(b_a_mix[rl_var_SV,:])
        sqrtC_noR, _, _, _ = calc_C_3D( \
                x_a, b_a, h, sqrtRinv, n_ob, n_e, \
                ln_var_obs, rl_var_obs, xi_obs)

    # Gradient of the costfunction
    gradJ = invIpC @ z - invsqrtIpC_t.T @ sqrtC.T @ sqrtRinv @ (y_mix - Hx_mix)
    gradJ += np.sum(sqrtP_f_mix[ln_var_SV,:] @ invsqrtIpC_t, axis = 0)
    gradJ += np.sum(sqrtP_f_mix[rl_var_SV,:] @ invsqrtIpC_t, axis = 0)
    gradJ -= np.sum(sqrtC_noR[ln_var_obs,:] @ invsqrtIpC_t, axis = 0)
    gradJ -= np.sum(sqrtC_noR[rl_var_obs,:] @ invsqrtIpC_t, axis = 0)

    return J, gradJ



def calc_C_3D(x_f, b_f, h, sqrtRinv, n_ob, n_e, \
    ln_var_obs, rl_var_obs, xi_obs):
    """
    Calculate the preconditioning matrices for the MLEF

    #### Input
    - `x_f`           ->  Background state, vector of size n_SV
    - `b_f`           ->  Perturbed state, array of size n_SV x n_e
    - `h`             ->  Observation operator, a function of the form
                            y = h(x)
    - `sqrtP_mix`     ->  Square root of the forecast error covariance matrix, 
                          array of size n_SV x n_e
    - `sqrtRinv`      ->  Inverse square root of the observation error covariance matrix, 
                          array of size n_obs x n_obs
    - `n_ob`          ->  Number of observed variables
    - `n_e`           ->  Number of ensemble members
    - `ln_var_SV`     ->  State variables that should be treated lognormally, 
                          as an array of indices between 0 and n_SV-1. 
    - `ln_var_obs`    ->  Observation variables that should be treated lognormally, 
                          as an array of indices between 0 and n_obs-1. 
    - `rl_var_SV`     ->  State variables that should be treated reverse lognormally, 
                          as an array of indices between 0 and n_SV-1. 
    - `rl_var_obs`    ->  Observation variables that should be treated reverse lognormally, 
                          as an array of indices between 0 and n_obs-1. 
    - `xi_SV`         ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`        ->  Parameter for the reverse lognormal distribution for the observations

    #### Output
    - `sqrtC`         ->  C^(1/2), array of size (n_obs n_obs_win) x n_e
    - `sqrtC_noR`     ->  H(x * b) - H(x), array of size n_obs x n_e
    - `invsqrtIpC_t`  ->  (I + C)^(-T/2), array of size n_e x n_e
    - `invIpC`        ->  (I + C)^(-1), array of size n_e x n_e
    """

    # h(b_f)
    h_b_f = np.array([ h(b_f[:,jE]) for jE in range(n_e) ]).T

    # Transform to mixed variables
    h_x_f_mix, h_b_f_mix, ln_var_obs, rl_var_obs \
        = transform_vars(h(x_f), h_b_f, ln_var_obs, rl_var_obs, xi_obs, ensembles = True)
    h_x_f_mix = np.broadcast_to(h_x_f_mix,(n_e,n_ob)).T

    # C**(1/2)
    sqrtC_noR = (h_b_f_mix - h_x_f_mix)
    sqrtC = sqrtRinv @ sqrtC_noR
    
    C = sqrtC.T @ sqrtC
    w, v = np.linalg.eigh(np.eye(n_e) + C)
    invsqrtIpC_t = v @ np.diag(w**(-1/2)) @ v.T
    invIpC = invsqrtIpC_t @ invsqrtIpC_t.T

    return sqrtC_noR, sqrtC, invsqrtIpC_t, invIpC

    


####################################################################################
####################################################################################
####################################################################################




def MLES(init_guess, t_obs, t_DA, n_t_mod, n_t_mod_obs, n_e, y, h, sqrtP_a, R, model, method='root'):
    """
    Maximum likelihood ensemble smoother

    Maximum likelihood ensemble smoother data assimilation technique for use in a general model.

    #### Input
    - `init_guess`  ->  Initial guess for the model analysis, vector of size n_SV
    - `t_obs`       ->  Time values of observations, vector of size n_t_obs
    - `t_DA`        ->  Time values of analysis window, vector of size n_win_DA
    - `n_t_mod`     ->  Number of time steps to use in each model run
    - `n_t_mod_obs` ->  Number of time steps to use in each model run between observation times
    - `n_e`         ->  Number of ensemble members
    - `y`           ->  Observations, array of size n_obs x n_t_obs
    - `h`           ->  Observation operator, a function of the form
                            y = h(x)
    - `sqrtP_a`     ->  Square root of the analysis error covariance matrix (initial), 
                        array of size n_SV x n_e
    - `R`           ->  Observation error covariance matrix, array of size n_obs x n_obs
                        This matrix is assumed to be constant throughout the assimilation
    - `model`       ->  Model to use in the analysis, function of the form
                            x = model(t,x_0),
                        where t contains the time values where the model is evaluated
                        and x_0 is the initial condition;
                        the output x are the state variables for all time values

    #### Output
    - `x_b`         ->  Background state, array of size n_SV x n_t
    - `x_a`         ->  Analysis state, array of size n_SV x n_t_obs
    - `t_true`      ->  Total time for background, vector of size n_t
    """

    # Save lengths of arrays
    n_SV = init_guess.size      # number of state variables
    n_ob, n_t_obs = y.shape     # number of observed variables and of observations
    n_win_DA = t_DA.size        # number of DA windows
    n_t = n_win_DA * n_t_mod + 1 # number of total time steps

    # Initialize background and analysis states
    x_b = np.empty((n_SV,n_t))
    x_a = np.empty((n_SV,n_win_DA))

    # At t=0, background state is set to the initial guess
    x_b[:,0] = init_guess
    sqrtP_f = sqrtP_a

    # Total time evolution for model
    dt_DA = t_DA[1] - t_DA[0]                           # Assuming evenly spaced DA windows
    t_true = np.linspace(t_DA[0],t_DA[-1] + dt_DA, n_t) # Assuming one prediction window
    
    # Inverse of observation error covariance matrix (constant throughout)
    sqrtRinv = cholesky(inv(R))

    # Loop over all DA windows
    for ii in range(n_win_DA):
        # Calculated states are on model time, and not DA time
        tt = ii * n_t_mod                # Index for model time at DA window
        tt_next = tt + n_t_mod           # Index for model time at next DA window
        sim_time = t_true[tt:tt_next+1]  # Simulation time of DA window

        # Find all observations within DA window
        ind_obs = np.nonzero(np.logical_and(sim_time[0] < t_obs, t_obs <= sim_time[-1]))[0]
        n_obs_win = len(ind_obs)
        if n_obs_win == 0:
            raise RuntimeError("No observations found in assimilation window!")
        y_win = y[:,ind_obs]
        t_obs_win = np.insert(t_obs[ind_obs], 0, sim_time[0])
        
        # Create long time vector for model run including observation times
        t_obs_long = np.empty(n_obs_win * n_t_mod_obs + 1)
        for j_o in range(n_obs_win):
            to = n_t_mod_obs*j_o
            t_obs_long[to:(to+n_t_mod_obs+1)] = np.linspace(t_obs_win[j_o],t_obs_win[j_o+1],n_t_mod_obs+1)

        ##################################################################
        ###################       Analysis step       ####################
        ##################################################################
        
        # Preconditioning
        sqrtC, invsqrtIpC_t, invIpC = calc_C_4D( \
            t_obs_long, model, h, x_b[:,tt], sqrtP_f, sqrtRinv, n_ob, n_obs_win, n_t_mod_obs, n_e)

        # Minimization of the cost function 
        tol = 1e-5
        if method == 'root':
            res = root( \
                grad_mles, \
                np.zeros(n_e), \
                args = (x_b[:,tt],y_win,h,sqrtP_f,invIpC,invsqrtIpC_t,sqrtC,sqrtRinv, \
                    model, t_obs_long, n_t_mod_obs), \
                jac = lambda *args : np.eye(n_e), \
                tol = tol, \
            )
            if res.nfev > 6:
                print("Number of iterations = "+str(res.nfev)+' at t = '+str(t_DA[ii]))
        if method == 'min':
            res = minimize( \
                cost_mles, \
                np.zeros(n_e), \
                args = (x_b[:,tt],y_win,h,sqrtP_f,invIpC,invsqrtIpC_t,sqrtC,sqrtRinv, \
                    model, t_obs_long, n_t_mod_obs), \
                jac = True, \
                tol = tol, \
                method = 'Newton-CG', \
                # method = 'BFGS', \
                # method = 'trust-exact', \
                hess = lambda *args : np.eye(n_e), \
                # options = {'disp': True}, \
            )
            if res.nit > 4:
                print("Number of iterations = "+str(res.nit)+' at t = '+str(t_DA[ii]))
        
        # Calculate and save analysis state
        x_a[:,ii] = x_b[:,tt] + sqrtP_f @ invsqrtIpC_t @ res.x

        ##################################################################
        ###################       Forecast step       ####################
        ##################################################################

        # Update square root analysis error covariance matrix
        _, invsqrtIpC_t, _ = calc_C_4D(\
            t_obs_long, model, h, x_a[:,ii], sqrtP_f, sqrtRinv, n_ob, n_obs_win, n_t_mod_obs, n_e)
        sqrtP_a = sqrtP_f @ invsqrtIpC_t

        # Model run for analysis state and square root errors (array of size n_e x n_SV x n_t)
        x_f = model(sim_time, x_a[:,ii])
        b_f = np.array([ model(sim_time, x_a[:,ii] + sqrtP_a[:,jE]) for jE in range(n_e) ])

        # Save background forecast
        x_b[:,tt+1:tt_next+1] = x_f[:,1:]

        # Square root forecast error covariance matrix
        sqrtP_f = b_f[:,:,-1].T - np.broadcast_to(x_f[:,-1],(n_e,n_SV)).T

        if not res.success:
            print("t = "+str(t_DA[ii])+": "+res.message)
            x_a[:,ii] = np.nan*np.empty_like(x_a[:,0])
    
    return x_a, x_b, t_true


def cost_mles(z, x_b, y, h, sqrtP_f, invIpC, invsqrtIpC_t, sqrtC, sqrtRinv, \
    model, t_obs_long, n_t_mod_obs):
    """
    Cost function for the MLES assimilation step, which needs to be minimized
    to obtain the optimal analysis state from the background and observations

    #### Input
    - `z`             ->  Preconditioned state to be optimized, vector of size n_e
    - `x_b`           ->  Background state, vector of size n_SV
    - `y`             ->  Observations, vector of size n_obs x n_obs_win
    - `h`             ->  Observation operator, a function of the form
                            y = h(x)
    - `sqrtP_f`       ->  Square root of the forecast error covariance matrix, 
                          array of size n_SV x n_e
    - `invIpC`        ->  (I + C)^(-1), array of size n_e x n_e
    - `invsqrtIpC_t`  ->  (I + C)^(-T/2), array of size n_e x n_e
    - `sqrtC`         ->  z = C^(1/2), array of size (n_obs n_obs_win) x n_e
    - `sqrtRinv`      ->  Inverse of the observation error covariance matrix, 
                          array of size n_obs x n_obs
    - `model`         ->  Model to use in the analysis, function of the form
                              x = model(t,x_0),
                          where t contains the time values where the model is evaluated
                          and x_0 is the initial condition;
                          the output x are the state variables for all time values
    - `t_obs_long`    ->  Time vector with observation times included, for forecasting
    - `n_t_mod_obs`   ->  Number of time steps to use in each model run between observation times

    #### Output
    - `J`             ->  Cost function for the MLES
    - `grad`          ->  Gradient of the cost function
    """

    # Get necessary shapes
    n_ob, n_obs_win = y.shape

    # Convert x_a from z for observations
    x_a = x_b + sqrtP_f @ invsqrtIpC_t @ z

    # Forecast to observation times
    x_obs_long = model(t_obs_long, x_a)
    x_obs = x_obs_long[:,::n_t_mod_obs]
    x_obs = x_obs[:,1:]                  # n_SV x n_obs_win

    # Calculate residual re = R^{-1/2} @ (y - Hx)
    re = np.empty(n_ob*n_obs_win)
    for j_ow in range(n_obs_win):
        j_o = n_ob*j_ow
        re[j_o:(j_o+n_ob)] = sqrtRinv @ (y[:,j_ow] - h(x_obs[:,j_ow]))

    # Cost function
    J = z @ invIpC @ z/2.0 + re.T @ re/2.0

    # Update sqrtC
    n_e = z.size
    sqrtC, _, _ = calc_C_4D(t_obs_long, model, h, x_a, sqrtP_f, sqrtRinv, n_ob, n_obs_win, n_t_mod_obs, n_e, x_obs_long)

    # Gradient of the costfunction
    gradJ = invIpC @ z - invsqrtIpC_t.T @ sqrtC.T @ re

    return J, gradJ

def grad_mles(z, x_b, y, h, sqrtP_f, invIpC, invsqrtIpC_t, sqrtC, sqrtRinv, \
    model, t_obs_long, n_t_mod_obs):
    """
    Cost function for the MLES assimilation step, which needs to be minimized
    to obtain the optimal analysis state from the background and observations

    #### Input
    - `z`             ->  Preconditioned state to be optimized, vector of size n_e
    - `x_b`           ->  Background state, vector of size n_SV
    - `y`             ->  Observations, vector of size n_obs x n_obs_win
    - `h`             ->  Observation operator, a function of the form
                            y = h(x)
    - `sqrtP_f`       ->  Square root of the forecast error covariance matrix, 
                          array of size n_SV x n_e
    - `invIpC`        ->  (I + C)^(-1), array of size n_e x n_e
    - `invsqrtIpC_t`  ->  (I + C)^(-T/2), array of size n_e x n_e
    - `sqrtC`         ->  z = C^(1/2), array of size (n_obs n_obs_win) x n_e
    - `sqrtRinv`      ->  Inverse of the observation error covariance matrix, 
                          array of size n_obs x n_obs
    - `model`         ->  Model to use in the analysis, function of the form
                              x = model(t,x_0),
                          where t contains the time values where the model is evaluated
                          and x_0 is the initial condition;
                          the output x are the state variables for all time values
    - `t_obs_long`    ->  Time vector with observation times included, for forecasting
    - `n_t_mod_obs`   ->  Number of time steps to use in each model run between observation times

    #### Output
    - `grad`          ->  Gradient of the cost function
    """

    # Get necessary shapes
    n_ob, n_obs_win = y.shape

    # Convert x_a from z for observations
    x_a = x_b + sqrtP_f @ invsqrtIpC_t @ z

    # Forecast to observation times
    x_obs_long = model(t_obs_long, x_a)
    x_obs = x_obs_long[:,::n_t_mod_obs]
    x_obs = x_obs[:,1:]                  # n_SV x n_obs_win

    # Calculate residual re = R^{-1/2} @ (y - Hx)
    re = np.empty(n_ob*n_obs_win)
    for j_ow in range(n_obs_win):
        j_o = n_ob*j_ow
        re[j_o:(j_o+n_ob)] = sqrtRinv @ (y[:,j_ow] - h(x_obs[:,j_ow]))

    # Update sqrtC
    n_e = z.size
    sqrtC, _, _ = calc_C_4D(t_obs_long, model, h, x_a, sqrtP_f, sqrtRinv, n_ob, n_obs_win, n_t_mod_obs, n_e, x_obs_long)

    # Gradient of the costfunction
    gradJ = invIpC @ z - invsqrtIpC_t.T @ sqrtC.T @ re

    return gradJ




def calc_C_4D(t_obs_long, model, h, x, sqrtP, sqrtRinv, n_ob, n_obs_win, n_t_mod_obs, n_e, x_obs_long = None):
    """
    Calculate the preconditioning matrices for the MLES

    #### Input
    - `t_obs_long`    ->  Time vector with observation times included, for forecasting
    - `model`         ->  Model to use in the analysis, function of the form
                              x = model(t,x_0),
                          where t contains the time values where the model is evaluated
                          and x_0 is the initial condition;
                          the output x are the state variables for all time values
    - `h`             ->  Observation operator, a function of the form
                            y = h(x)
    - `x`             ->  Background state, vector of size n_SV
    - `sqrtP`         ->  Square root of the forecast error covariance matrix, 
                          array of size n_SV x n_e
    - `sqrtRinv`      ->  Inverse of the observation error covariance matrix, 
                          array of size n_obs x n_obs
    - `n_ob`          ->  Number of observed variables
    - `n_obs_win`     ->  Number of observations in the window
    - `n_t_mod_obs`   ->  Number of time steps to use in each model run between observation times
    - `n_e`           ->  Number of ensemble members

    #### Output
    - `sqrtC`         ->  C^(1/2), array of size (n_obs n_obs_win) x n_e
    - `invsqrtIpC_t`  ->  (I + C)^(-T/2), array of size n_e x n_e
    - `invIpC`        ->  (I + C)^(-1), array of size n_e x n_e
    """

    # Forecast to all observations
    if x_obs_long is None:
        x_obs_long = model(t_obs_long, x)
    b_obs_long = np.array([ model(t_obs_long, x + sqrtP[:,jE]) for jE in range(n_e) ])

    # Select variables at observation times
    x_obs = x_obs_long[:,::n_t_mod_obs]
    x_obs = x_obs[:,1:]                     # n_SV x n_obs_win
    b_obs = b_obs_long[:,:,::n_t_mod_obs]
    b_obs = b_obs[:,:,1:]                   # n_e x n_SV x n_obs_win

    # Apply observation operator
    sqrtC = np.empty((n_ob*n_obs_win,n_e))
    for j_ow in range(n_obs_win):
        j_o = n_ob*j_ow
        h_x_obs = np.broadcast_to(h(x_obs[:,j_ow]),(n_e,n_ob)).T          # n_ob x n_e
        h_b_obs = np.array([h(b_obs[jE,:,j_ow]) for jE in range(n_e)]).T  # n_ob x n_e

        sqrtC[j_o:(j_o+n_ob),:] = sqrtRinv @ (h_b_obs - h_x_obs)
    
    # Calculate C, (I + C)^(-T/2), and (I + C)^(-1) for preconditioning
    C = sqrtC.T @ sqrtC
    w, v = np.linalg.eigh(C)
    invsqrtIpC_t = v @ np.diag((1 + w)**(-1/2)) @ v.T
    invIpC = invsqrtIpC_t @ invsqrtIpC_t.T

    return sqrtC, invsqrtIpC_t, invIpC