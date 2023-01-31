""" 
Functions needed for solving the Lorenz 63 model

#### Methods:
- `sol_l63`         ->  Solution to the Lorenz-63 model
- `model_l63`       ->  Lorenz-63 model for specific use in DA modules
- `sol_l63_tlm`     ->  Solution to the Lorenz-63 model with tangent linear model
- `model_l63_tlm`   ->  Lorenz-63 model for specific use in DA modules with tangent linear model

- `create_B_init`   ->  Create B-matrix based on L63 run
- `nature_run_obs`  ->  Nature run + generate observations in L63 model

#### Author:
Senne Van Loon
Cooperative Institute for Research in the Atmosphere (CIRA),
Colorado State University,
3925A West Laporte Ave, Fort Collins, CO 80521

#### References and acknowledgements:
[1] Fletcher, S. J. (2017). Data assimilation for the geosciences: From theory to application. Elsevier
[2] Goodliff, M., Fletcher, S., Kliewer, A., Forsythe, J., & Jones, A. (2020). Detection of non-Gaussian behavior using machine learning techniques: A case study on the Lorenz 63 model. Journal of Geophysical Research: Atmospheres, 125, e2019JD031551. https://doi.org/10.1029/2019JD031551
[3] Goodliff, M. R., Fletcher, S. J., Kliewer, A. J., Jones, A. S., & Forsythe, J. M. (2022). Non-Gaussian detection using machine learning with data assimilation applications. Earth and Space Science, 9, e2021EA001908. https://doi.org/10.1029/2021EA001908
[4] Based on code from Goodliff, M. used in [2] and [3]
[5] Based on code from Hossen, J.

"""

import numpy as np
from scipy.integrate import solve_ivp

import sys
sys.path.append("../DataAssimilation/")
from mod_DA_general import getBsimple


def l63_fun(t, SV, s, r, b):
    """
    Equations of the Lorenz 63 model, vectorized
    """

    # State variables
    x = SV[0,:]
    y = SV[1,:]
    z = SV[2,:]

    dx_dt = -s*(x-y)
    dy_dt = r*x-y-z*x
    dz_dt = x*y-b*z

    dSV_dt = np.array([dx_dt,dy_dt,dz_dt])

    if SV.shape == dSV_dt.shape:
        return dSV_dt
    else:
        print('Input and output sizes are not equal!')
        return None

def l63_fun_SV(t, SV, s, r, b):
    """
    Equations of the Lorenz 63 model
    """

    # State variables
    x = SV[0]
    y = SV[1]
    z = SV[2]

    dx_dt = -s*(x-y)
    dy_dt = r*x-y-z*x
    dz_dt = x*y-b*z

    return np.array([dx_dt,dy_dt,dz_dt])

def sol_l63(t_span, \
        SV_init = np.array([-3.12346395,-3.12529803,20.69823159]), \
        p = np.array([10.0,28.0,8.0/3.0]), \
        t_eval = None, \
        meth = 'RK45'):
    """
    Solution to the Lorenz 63 model 
    
    #### Input
    - `t_span`  ->  Beginning and endpoints in time
    - `SV_init` ->  Initial value for all state variables, array of size n_SV
    - `p`       ->  Parameters of the Lorenz model, [sigma, rho, beta]
    - `t_eval`  ->  Time points where the model is evaluated
    - `meth`    ->  String indicating the integration method to use

    #### Output
    - `t`       ->  Time vector of size n_t
    - `y`       ->  State variables at every time point, array of size n_SV x n_t
    """

    # Solve the Lorenz 63 model
    if meth.casefold() == 'euler':
        # Define time vector
        if np.all(t_eval == None):
            dt = 0.01
            t = np.arange(t_span[0],t_span[1],dt)
        else:
            dt= t_eval[1]-t_eval[0]
            t = t_eval

        # Initialize state variables
        SV = np.zeros((3,t.size))
        SV[:,0] = SV_init
        for ii in range(t.size-1):
            SV[:,ii+1] = SV[:,ii] + l63_fun_SV(t[ii], SV[:,ii], p[0],p [1], p[2]) * dt
        
        return t, SV
    else:
        sol = solve_ivp( \
            l63_fun, \
            t_span, \
            SV_init, \
            method = meth, \
            t_eval = t_eval, \
            vectorized = True, \
            args = (p[0],p[1],p[2]) \
        )

        # Return none if integration was unsuccessful
        if not sol.success:
            return None, None 

        return sol.t, sol.y


def model_l63(t, x0):
    """
    L63 model run for input in data assimilation methods
    """

    p_L63 = np.array([10.0,28.0,8.0/3.0])
    _, y = sol_l63([t[0],t[-1]], x0, p_L63, t, meth='RK45')

    return y

###########################################################################################
###############################    Tangent linear model     ###############################
###########################################################################################

def l63_fun_tlm(t, SV, s, r, b):
    """
    Equations of the Lorenz 63 model with the tangent linear model
    """

    # State variables
    x = SV[0,:]
    y = SV[1,:]
    z = SV[2,:]
    M = np.reshape(SV[3:],(3,3))

    dx_dt = -s*(x-y)
    dy_dt = r*x-y-z*x
    dz_dt = x*y-b*z

    zrs = np.zeros_like(x)
    F = np.array([ \
        [-s+zrs, s+zrs , zrs   ], \
        [r-z   , -1+zrs, -x    ], \
        [y     , x     , -b+zrs]  \
    ])

    dM_dt_M = np.dot(np.transpose(F,[2,0,1]), M)
    dM_dt = dM_dt_M.reshape((-1,9)).T

    dSV_dt = np.concatenate([np.array([dx_dt,dy_dt,dz_dt]),dM_dt])

    if SV.shape == dSV_dt.shape:
        return dSV_dt
    else:
        print('Input and output sizes are not equal!')
        return None


def sol_l63_tlm(t_span, \
        SV_init = np.array([-3.,-3.,20.,1.,0.,0.,0.,1.,0.,0.,0.,1.]), \
        p = np.array([10.0,28.0,8.0/3.0]), \
        t_eval = None, \
        meth = 'RK45'):
    """
    Solution to the Lorenz 63 model with tangent linear model
    
    #### Input
    - `t_span`  ->  Beginning and endpoints in time
    - `SV_init` ->  Initial value for all state variables, array of size n_SV
    - `p`       ->  Parameters of the Lorenz model, [sigma, rho, beta]
    - `t_eval`  ->  Time points where the model is evaluated
    - `meth`    ->  String indicating the integration method to use

    #### Output
    - `t`       ->  Time vector of size n_t
    - `y`       ->  State variables at every time point, array of size n_SV x n_t
    - `M`       ->  Tangent linear model at every time point, array of size n_SV x n_SV x n_t
    """

    # Solve the Lorenz 63 model
    sol = solve_ivp( \
        l63_fun_tlm, \
        t_span, \
        SV_init, \
        method = meth, \
        t_eval = t_eval, \
        vectorized = True, \
        args = (p[0],p[1],p[2]) \
    )

    # Return none if integration was unsuccessful
    if not sol.success:
        return None, None 

    y = sol.y[:3,:]
    M = sol.y[3:,:].reshape((3,3,-1))
    return sol.t, y, M

def model_l63_tlm(t, x0):
    """
    L63 model run with tangent linear model for input in data assimilation methods
    """

    p_L63 = np.array([10.0,28.0,8.0/3.0])

    M0 = np.eye(3).reshape(-1)
    x0_M0 = np.concatenate([x0,M0])

    _, y, M = sol_l63_tlm([t[0],t[-1]], x0_M0, p_L63, t, meth='RK45')

    return y, M





###########################################################################################
#######################    Data assimilation specific functions     #######################
###########################################################################################

def create_B_init( \
        SV_init = np.array([-10.0,-10.0,20.0]), \
        p = np.array([10.0,28.0,8.0/3.0]), \
        meth = 'RK45'):
    """
    Create the initial guess for the background error covariance matrix for the Lorenz 63 model. 

    #### Optional input
    - `SV_init` ->  Initial value for all state variables, array of size n_SV
    - `p`       ->  Parameters of the Lorenz model, [sigma, rho, beta]
    - `meth`    ->  String indicating the integration method to use

    #### Output
    - `B`       ->  Background error covariance matrix
    """

    # Create time values for evaluation
    total_steps = 2000
    dt = 0.01
    t_span = [0.0,dt*total_steps]
    t_eval = np.arange(t_span[0],t_span[1],dt)

    # Solve Lorenz model
    _, SV = sol_l63(t_span,SV_init,p,t_eval,meth)

    # Calculate the background error covariance matrix 
    B = getBsimple(SV)

    return B


def create_H(obsgrid='xyz'):
    """
    Create the observation operator H

    #### Input
    - `obsgrid` ->  String with observed state variables in alphabetical order

    #### Output
    - `H`       ->  Observation operator
    """

    # Initialize matrix
    Nx = 3
    Ny = len(obsgrid)
    H = np.zeros((Ny, Nx))

    if Ny > 3:
        print('Input string obsgrid should have length smaller than 3!')
        return None

    # Select observed variables
    if obsgrid == 'x':
        observed_vars = [0]
    elif obsgrid == 'y':
        observed_vars = [1]
    elif obsgrid == 'z':
        observed_vars = [2]
    elif obsgrid == 'xy':
        observed_vars = [0, 1]
    elif obsgrid == 'xz':
        observed_vars = [0, 2]
    elif obsgrid == 'yz':
        observed_vars = [1, 2]
    elif obsgrid == 'xyz':
        observed_vars = [0, 1, 2]
    else:
        observed_vars = [0, 1]
    
    # Fill in ones at desired places
    for ii in range(Ny):
        H[ii, observed_vars[ii]] = 1.0
    
    return H
