""" 
Functions needed for Kalman-based data assimilation techniques,
including the maximum likelihood ensemble filter

#### Methods:
- `kalman_filter`       ->  Kalman filter
- `kalman_filter_gauss` ->  Gaussian Kalman filter

#### Author:
Senne Van Loon
Cooperative Institute for Research in the Atmosphere (CIRA),
Colorado State University,
3925A West Laporte Ave, Fort Collins, CO 80521

#### References and acknowledgements:
[1] Fletcher, S. J. (2017). Data assimilation for the geosciences: From theory to application. Elsevier
[2] Steven J. Fletcher, Milija Zupanski, Michael R. Goodliff, Anton J. Kliewer, Andrew S. Jones, John M. Forsythe, Ting-Chi Wu, Md. Jakir Hossen, and Senne van Loon. Lognormal and Mixed Gaussian-Lognormal Kalman Filters

"""

# Load modules
import numpy as np
from scipy.linalg import inv
from mod_DA_general import get_ln_rl_var, transform_vars, time_limit, TimeoutException

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)


def kalman_filter(init_guess, t_obs, n_t_mod, y, H, P_a, R, Q, model, \
        ln_vars_SV = [], ln_vars_obs = [], \
        rl_vars_SV = [], rl_vars_obs = [], xi_SV = 0.0, xi_obs = 0.0):
    """
    Kalman filter

    Kalman filter data assimilation technique for use in a general model, 
    which allows for mixed Gaussian, lognormal, and reverse lognormal observations 
    and background state variables

    #### Input
    - `init_guess`  ->  Initial guess for the model analysis, vector of size n_SV
    - `t_obs`       ->  Time values of observations, vector of size n_t_obs
    - `n_t_mod`     ->  Number of time steps to use in each model run
    - `y`           ->  Observations, array of size n_obs x n_t_obs
    - `H`           ->  Observation operator, obtained from create_H, 
                        array of size n_obs x n_SV
    - `P_a`         ->  Analysis error covariance matrix (initial), array of size n_SV x n_SV
    - `R`           ->  Observation error covariance matrix, array of size n_obs x n_obs
    - `Q`           ->  Model error covariance matrix, array of size n_SV x n_SV
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
    _, n_SV = H.shape           # number of state variables
    n_t_obs = t_obs.size        # number of observations
    n_t = n_t_obs * n_t_mod + 1 # number of total time steps

    # Initialize background, analysis, and "true" states
    x_b = np.empty((n_SV,n_t))
    x_a = np.empty((n_SV,n_t_obs))
    x_t = np.empty((n_SV,n_t_obs))
    e_a_mix = np.sqrt(np.diagonal(P_a))

    # At t=0, background and analysis states are set to the initial guess
    x_b[:,0] = init_guess
    x_a[:,0] = init_guess

    # "True" state at t = 0
    ln_var_SV,  rl_var_SV  = get_ln_rl_var(ln_vars_SV, rl_vars_SV, n_t_obs,0,x_a[:,0])
    e_a = e_a_mix
    e_a[ln_var_SV] = np.exp(e_a_mix[ln_var_SV])
    e_a[rl_var_SV] = xi_SV - np.exp(e_a_mix[rl_var_SV])
    x_t[:,0] = x_a[:,0] - e_a
    x_t[ln_var_SV,0] = x_a[ln_var_SV,0] / e_a[ln_var_SV] 
    x_t[rl_var_SV,0] = xi_SV - (xi_SV - x_a[rl_var_SV,0]) / (xi_SV - e_a[rl_var_SV]) 

    # Total time evolution for model
    dt_obs = t_obs[1] - t_obs[0] # Assuming evenly spaced observations
    t_true = np.linspace(t_obs[0],t_obs[-1] + dt_obs, n_t) # Assuming one prediction window

    timeout=False
    # Loop over all observations
    for ii in range(1,n_t_obs):
        # Calculated states are on model time, and not observation time
        tt = ii * n_t_mod       # Index for model time at observation point
        tt_prev = tt - n_t_mod  # Index for model time at previous observation point
        sim_time = t_true[tt_prev:tt+1]

        ##################################################################
        ###################       Forecast step       ####################
        ##################################################################
        
        # Model forecast
        try:
            with time_limit(60):
                x_f = model(sim_time, x_a[:,ii-1])
                b_f = model(sim_time, x_t[:,ii-1])  # "True state" at current time
                if timeout:
                    x_a[:,ii-1] = np.nan*np.empty(n_SV)
                    x_t[:,ii-1] = np.nan*np.empty(n_SV)
                    timeout = False
        except TimeoutException:
            print("Warning: timeout in model run, skipping this assimilation step!")
            print("    x_a_init = "+str(x_a[:,ii-1]))
            print("    x_t_init = "+str(x_t[:,ii-1]))

            # If timeout, something went wrong in previous analysis
            x_a[:,ii-1] = np.nan*np.empty(n_SV)
            x_b[:,tt_prev:tt+1] = np.nan*np.empty_like(sim_time)

            # Restart DA from initial guess next step
            x_a[:,ii] = init_guess
            x_t[:,ii] = x_a[:,ii] - e_a
            x_t[ln_var_SV,ii] = x_a[ln_var_SV,ii] / e_a[ln_var_SV] 
            x_t[rl_var_SV,ii] = xi_SV - (xi_SV - x_a[rl_var_SV,ii]) / (xi_SV - e_a[rl_var_SV]) 
            timeout = True
            continue

        # Save background forecast
        x_b[:,tt_prev+1:tt+1] = x_f[:,1:]

        # Get indices for lognormally and reverse lognormally distributed state variables 
        # and observations for this time step
        ln_var_SV,  rl_var_SV  = get_ln_rl_var(ln_vars_SV, rl_vars_SV, n_t_obs,ii,x_f[:,-1])
        ln_var_obs, rl_var_obs = get_ln_rl_var(ln_vars_obs,rl_vars_obs,n_t_obs,ii,y[:,ii])

        # Transform forecast and observations to mixed variables
        Hx = H @ x_f[:,-1]
        x_f_mix, b_f_mix, ln_var_SV, rl_var_SV\
            = transform_vars(x_f[:,-1], b_f[:,-1], ln_var_SV, rl_var_SV, xi_SV)
        y_mix, Hx_mix, ln_var_obs, rl_var_obs \
            = transform_vars(y[:,ii], Hx, ln_var_obs, rl_var_obs, xi_obs)
        
        # Forecast error and error covariance matrix
        e_f_mix = x_f_mix - b_f_mix
        P_f = np.outer(e_f_mix,e_f_mix) + Q

        # Gradient matrices
        W_b, W_o_inv = calc_Wb_invWo(x_f[:,-1], H, ln_var_SV, rl_var_SV, ln_var_obs, rl_var_obs, xi_SV, xi_obs)

        # Scaled observation operator
        H_til = W_o_inv @ H @ W_b
        
        ##################################################################
        ####################       Update step       #####################
        ##################################################################

        # Kalman gain matrix (size n_SV x n_obs)
        K = P_f @ H_til.T @ inv(H_til @ P_f @ H_til.T + R)

        # Analysis state
        x_a_mix = x_f_mix + K @ (y_mix - Hx_mix)

        # Analysis error covariance matrix
        # P_a = (np.eye(n_SV) - K @ H_til) @ P_f @ np.transpose(np.eye(n_SV) - K @ H_til) + K @ R @ K.T
        P_a = (np.eye(n_SV) - K @ H_til) @ P_f
        try:
            e_a_mix = np.sqrt(np.diagonal(P_a))
        except:
            print("Warning: analysis error covariance matrix has negative elements on its diagonal!")
            e_a_mix = np.sqrt(np.abs(np.diagonal(P_a)))

        # Transform back to normal variables
        x_a[:,ii] = x_a_mix
        x_a[ln_var_SV,ii] = np.exp(x_a[ln_var_SV,ii])
        x_a[rl_var_SV,ii] = xi_SV - np.exp(x_a[rl_var_SV,ii])

        # Transform mixed errors for this time step
        e_a = e_a_mix
        e_a[ln_var_SV] = np.exp(e_a_mix[ln_var_SV])
        e_a[rl_var_SV] = xi_SV - np.exp(e_a_mix[rl_var_SV])

        # "True state" for this time step
        x_t[:,ii] = x_a[:,ii] - e_a
        x_t[ln_var_SV,ii] = x_a[ln_var_SV,ii] / e_a[ln_var_SV] 
        x_t[rl_var_SV,ii] = xi_SV - (xi_SV - x_a[rl_var_SV,ii]) / (xi_SV - e_a[rl_var_SV]) 

    # Final forecasting step
    ii += 1
    tt = ii * n_t_mod       # Index for model time at observation point
    tt_prev = tt - n_t_mod  # Index for model time at previous observation point
    sim_time = t_true[tt_prev:tt+1]

    # Model run
    try:
        with time_limit(60):
            x_f = model(sim_time, x_a[:,ii-1])
            x_b[:,tt_prev+1:tt+1] = x_f[:,1:]
    except TimeoutException:
        # Save model run in background
        x_b[:,tt_prev:tt+1] = np.nan*np.empty_like(sim_time)
    
    return x_a, x_b, t_true




def kalman_filter_gauss(init_guess, t_obs, n_t_mod, y, H, P_a, R, Q, model_tlm):
    """
    Gaussian Kalman filter

    Kalman filter data assimilation technique for use in a general model 
    for Gaussian distributed observations and background state variables

    #### Input
    - `init_guess`  ->  Initial guess for the model analysis, vector of size n_SV
    - `t_obs`       ->  Time values of observations, vector of size n_t_obs
    - `n_t_mod`     ->  Number of time steps to use in each model run
    - `y`           ->  Observations, array of size n_obs x n_t_obs
    - `H`           ->  Observation operator, obtained from create_H, 
                        array of size n_obs x n_SV
    - `P_a`         ->  Analysis error covariance matrix (initial), array of size n_SV x n_SV
    - `R`           ->  Observation error covariance matrix, array of size n_obs x n_obs
                        This matrix is assumed to be constant throughout the assimilation
    - `Q`           ->  Model error covariance matrix, array of size n_SV x n_SV
    - `model_tlm`   ->  Model to use in the analysis, function of the form
                            x, M = model_tlm(t,x_0),
                        where t contains the time values where the model is evaluated
                        and x_0 is the initial condition;
                        the output x are the state variables for all time values,
                        and the tangent linear model matrix M

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

    # Total time evolution for model
    dt_obs = t_obs[1] - t_obs[0] # Assuming evenly spaced observations
    t_true = np.linspace(t_obs[0],t_obs[-1] + dt_obs, n_t) # Assuming one prediction window

    # Loop over all observations
    for ii in range(1,n_t_obs):
        # Calculated states are on model time, and not observation time
        tt = ii * n_t_mod       # Index for model time at observation point
        tt_prev = tt - n_t_mod  # Index for model time at previous observation point
        sim_time = t_true[tt_prev:tt+1]

        ##################################################################
        ###################       Forecast step       ####################
        ##################################################################
        
        # Model run forecast and tangent linear model
        x_f, M_f = model_tlm(sim_time, x_a[:,ii-1])

        # Save background forecast
        x_b[:,tt_prev+1:tt+1] = x_f[:,1:]

        # Update forecast error covariance matrix
        P_f = M_f[:,:,-1] @ P_a @ M_f[:,:,-1].T + Q
        
        ##################################################################
        ####################       Update step       #####################
        ##################################################################

        # Kalman gain matrix
        K = P_f @ H.T @ inv(H @ P_f @ H.T + R)

        # Analysis error covariance
        # P_a = (np.eye(n_SV) - K @ H) @ P_f @ np.transpose(np.eye(n_SV) - K @ H) + K @ R @ K.T
        P_a = (np.eye(n_SV) - K @ H) @ P_f

        # Analysis state
        x_a[:,ii] = x_f[:,-1] + K @ (y[:,ii] - H @ x_f[:,-1])

    # Final forecasting step
    ii += 1
    tt = ii * n_t_mod       # Index for model time at observation point
    tt_prev = tt - n_t_mod  # Index for model time at previous observation point
    sim_time = t_true[tt_prev:tt+1]

    # Model run
    x_f, _ = model_tlm(sim_time, x_a[:,ii-1])
    x_b[:,tt_prev+1:tt+1] = x_f[:,1:]
    
    return x_a, x_b, t_true









def calc_Wb_invWo(SV, H, ln_var_SV, rl_var_SV, ln_var_obs, rl_var_obs, xi_SV, xi_obs):
    """
    Calculate the gradient matrices W_b and W_o^-1

    #### Input
    - `SV`          ->  State variables to use in the matrices, array of size n_SV
    - `H`           ->  Observation operator, either
                            * array of size n_obs x n_SV,  
                            * function of the form y = h(SV)
    - `ln_var_SV`   ->  State variables that should be treated lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `ln_var_obs`  ->  Observation variables that should be treated lognormally, 
                        as an array of indices between 0 and n_obs-1. 
    - `rl_var_SV`   ->  State variables that should be treated reverse lognormally, 
                        as an array of indices between 0 and n_SV-1. 
    - `rl_var_obs`  ->  Observation variables that should be treated reverse lognormally, 
                        as an array of indices between 0 and n_obs-1. 
    - `xi_SV`       ->  Parameter for the reverse lognormal distribution for the state variables
    - `xi_obs`      ->  Parameter for the reverse lognormal distribution for the observations

    #### Output
    - `W_b`         ->  Gradient matrix W_b = (dX(x)/dx)^-1, diagonal matrix of size n_SV x n_SV,
                            W_b[gs_vars_SV] = 1
                            W_b[ln_vars_SV] = x
                            W_b[rl_vars_SV] = x - xi_SV
    - `invW_o`      ->  Gradient matrix W_o^-1 = dH(h)/dh, diagonal matrix of size n_obs x n_obs,
                            W_b[gs_vars_obs] = 1
                            W_b[ln_vars_obs] = 1/h(x)
                            W_b[rl_vars_obs] = 1/(h(x) - xi_obs)
    """

    # number of observed and total state variables

    # Initialize observations array
    if callable(H):
        Hx = H(SV)
    else:
        Hx = H @ SV
    n_SV  = SV.size
    n_obs = Hx.size

    # W_b = (dX(x)/dx)^-1
    W_b = np.eye(n_SV)
    W_b[ln_var_SV,ln_var_SV] = SV[ln_var_SV]
    W_b[rl_var_SV,rl_var_SV] = SV[rl_var_SV] - xi_SV
    
    # W_o^-1 = dH(h)/dh
    W_o_inv = np.eye(n_obs)
    W_o_inv[ln_var_obs,ln_var_obs] = 1.0/Hx[ln_var_obs]
    W_o_inv[rl_var_obs,rl_var_obs] = 1.0/(Hx[rl_var_obs] - xi_obs)

    return W_b, W_o_inv