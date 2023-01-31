""" 
Functions needed for general data assimilation techniques

#### Methods:
- `getBsimple`     ->  First guess for error covariance matrix
- `getBupdate`     ->  Updates for the error covariance matrix
- `gen_obs`        ->  Generation of observations from true state
- `rmse`           ->  Root mean square error of all variables
- `rmse_time`      ->  Root mean square error averaged over time
- `get_ln_rl_var`  ->  Get the correct distribution to be used for a certain time step
- `transform_vars` ->  Transform state variables to mixed representation

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

# Load modules
import numpy as np
from scipy.stats import rv_continuous
from scipy.special import erf, erfinv
import copy

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

class reverse_lognormal(rv_continuous):
    "Reverse lognormal distribution"
    # Probability density function
    def _pdf(self, x, mu = 0.0, sigma = 1.0, T = 0.0):
        return np.exp(-(np.log(T-x) - mu)**2/2/sigma**2)/(T-x)/sigma/np.sqrt(2*np.pi)

    # Cumulative probability density function
    def _cdf(self, x, mu = 0.0, sigma = 1.0, T = 0.0):
        return (1.0-erf((np.log(T-x) - mu)/np.sqrt(2)/sigma))/2.0

    # Complement of the cumulative probability density function
    def _sf(self, x, mu = 0.0, sigma = 1.0, T = 0.0): 
        return (1.0+erf((np.log(T-x) - mu)/np.sqrt(2)/sigma))/2.0

    # Inverse of the cumulative probability density function
    def _ppf(self, p, mu = 0.0, sigma = 1.0, T = 0.0):
        return T - np.exp(np.sqrt(2) * sigma * erfinv(1.0-2.0*p) + mu)

    # T and mu can be any value between -oo and +oo
    def _argcheck(self, mu = 0.0, sigma = 1.0, T = 0.0):
        return (sigma > 0)

    # Support of the probability (max = T)
    def _get_support(self, mu = 0.0, sigma = 1.0, T = 0.0):
        return -np.inf, T

def getBsimple(SV):
    """
    Obtain the first guess for the background error covariance matrix.

    #### Input
    - SV  ->  state variables of model, should be of size n_SV x n_t,
            with n_SV the number of variables, and n_t the number of time steps

    #### Output
    - B   ->  Background error covariance matrix of size n_SV x n_SV
    """

    # Sample frequency and largest error normalization
    sample_freq = 16
    err2 = 2

    # Create sample of state variables
    SV_sample = SV[:,::sample_freq]

    # Create background error covariance matrix from sample
    B = np.cov(SV_sample)

    # Scale background error covariance matrix
    alpha = err2/np.amax(np.diag(B))
    B = alpha*B
    
    return B

def getBupdate(x_a, x_b, ln_vars = [], rl_vars = [], xi_SV = 0.0):
    """
    Update the background error covariance matrix with updated background and current trajectories
                  `B = Sum_i^S <x_b - x_a, (x_b - x_a)^T >/S`

    #### Input
    - `x_a`      ->   Analysis trajectory, array with size n_SV x n_t. 
    - `x_b`      ->   Background trajectory, array with size n_SV x n_t,
    - `ln_vars`  ->   State variables that should be treated lognormally, 
                    as an array of indices between 0 and n_SV-1. 
                    Default is `[]` for all Gaussian variables
    - `rl_vars`  ->   State variables that should be treated reverse lognormally, 
                    as an array of indices between 0 and n_SV-1. 
                    Default is `[]` for all Gaussian variables
    - `xi_SV`    ->   Parameter for the reverse lognormal distribution
    Output:
    - `B`        ->   Updated background error covariance matrix of size n_SV x n_SV
    """

    # Total time 
    _, n_t = x_a.shape
    
    # Matrix with trajectory values for Gaussian covariance
    X = x_a - x_b

    # Use logaritmic variables for lognormally distributed state variables
    if ln_vars:
        try:
            X[ln_vars, :] = np.log(x_a[ln_vars, :]) - np.log(x_b[ln_vars, :])
        except RuntimeWarning:
            # Use Gaussian variables anyway when either x_a or x_b is negative
            pass


    # Use reverse logaritmic variables for reverse lognormally distributed state variables
    if rl_vars:
        try:
            X[rl_vars, :] = np.log(xi_SV - x_a[rl_vars, :]) - np.log(xi_SV - x_b[rl_vars, :])
        except RuntimeWarning:
            # Use Gaussian variables anyway when either x_a or x_b is larger than xi_SV
            pass

    # Updated background error covariance matrix
    B = (X @ X.T)/n_t

    return B

def gen_obs(t, SV, period_obs, H, var_obs, ln_vars = [], rl_vars = [], xi_obs = 0.0, \
            seed = None, sample = 'mode'):
    """
    Generate noisy observations from true state nature run

    #### Input
    - `t`               ->  Time of truth, vector of length n_t
    - `SV`              ->  State variables of truth, array of size n_SV x n_t,
                            with n_SV the number of variables, and n_t the number of time steps
    - `period_obs`      ->  Observation period, in steps of truth time
    - `H`               ->  Observation operator, either
                                * array of size n_obs x n_SV,  
                                * function of the form y = h(SV)
    - `var_obs`         ->  Observational error variance, diagonal values of 
                            observational error covariance matrix R
    - `ln_vars`         ->  State variables that should be treated lognormally, 
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
    - `rl_vars`         ->  State variables that should be treated reverse lognormally, same as ln_vars
    - `xi_obs`          ->  Parameter for the reverse lognormal distribution
    - `seed`            ->  Seed of the random number generator. Default is random seed
    - `sample`          ->  Descriptor to sample around, can be either `mode`, `median`, or `mean`. 
                            Default is `mode`.

    #### Output 
    - `t_obs`           ->  Time of observations, vector of length n_t_obs = n_t//period_obs
    - `y`               ->  Observations, array of size n_obs x n_t_obs
    - `R`               ->  Observational error covariance matrix of size n_obs x n_obs
    """

    # Count lognormal and reverse lognormal observations
    n_ln, n_rl = 0, 0

    # Create time array of the observations
    t_obs = t[::period_obs]
    n_t_obs = t_obs.size

    # State variables at observation times
    SV_obs = SV[:,::period_obs]

    # Initialize observations array
    if callable(H):
        y = H(SV_obs)
    else:
        y = H @ SV_obs 
    n_obs, _ = y.shape

    # Create observational error covariance matrix 
    R = var_obs*np.eye(n_obs)

    # Create random number generator
    rng = np.random.default_rng(seed)
    rln = reverse_lognormal(shapes = 'mu, sigma, T', seed = seed)

    # Loop over observations
    for ii in range(1,n_t_obs):

        # Get indices for lognormally and reverse lognormally distributed state variables 
        # and observations for this time step
        ln_var, rl_var = get_ln_rl_var(ln_vars, rl_vars, n_t_obs, ii, SV_obs[:,ii])

        # Add noise
        for jj in range(n_obs):
            y0 = y[jj,ii]
            r0 = R[jj,jj]
            if jj in ln_var: # Lognormal noise

                # Calculate mu and sigma for given descriptor
                if sample == 'mean':
                    ln_mu = np.log(y0**2/np.sqrt(r0+y0**2))
                    ln_sd = np.sqrt(np.log(1.0 + r0/y0**2))
                if sample == 'median':
                    ln_mu = np.log(y0)
                    ln_sd = np.sqrt(np.log(0.5 + 0.5*np.sqrt(4.0*r0/y0**2 + 1)))
                if sample == 'mode':
                    p = np.polynomial.polynomial.Polynomial((-r0/y0**2,0,0,-1.0,1.0))
                    rts = p.roots()
                    realRoot = np.real(rts[np.logical_and(np.imag(rts) < 1e-5, np.real(rts) > 1.0)])
                    if not realRoot.size == 1 :
                        raise RuntimeError("Multiple roots found while solving for the mode variance!")
                    ln_mu = np.log(y0*realRoot)
                    ln_sd = np.sqrt(np.log(realRoot))
                
                # Sample from lognormal distribution
                y[jj,ii] = rng.lognormal(ln_mu, ln_sd)
                n_ln += 1
            elif jj in rl_var:  # Reverse lognormal noise

                # Calculate mu and sigma for given descriptor
                if sample == 'mean':
                    rl_mu = np.log((xi_obs-y0)**2/np.sqrt(r0+(xi_obs-y0)**2))
                    rl_sd = np.sqrt(np.log(1.0 + r0/(xi_obs-y0)**2))
                if sample == 'median':
                    rl_mu = np.log(xi_obs-y0)
                    rl_sd = np.sqrt(np.log(0.5 + 0.5*np.sqrt(4.0*r0/(xi_obs-y0)**2 + 1)))
                if sample == 'mode':
                    p = np.polynomial.polynomial.Polynomial((-r0/(xi_obs-y0)**2,0,0,-1.0,1.0))
                    rts = p.roots()
                    realRoot = np.real(rts[np.logical_and(np.imag(rts) < 1e-5, np.real(rts) > 1.0)])
                    if not realRoot.size == 1 :
                        raise RuntimeError("Multiple roots found while solving for the mode variance!")
                    rl_mu = np.log((xi_obs-y0)*realRoot)
                    rl_sd = np.sqrt(np.log(realRoot))

                # Sample from reverse lognormal distribution
                y[jj,ii] = rln.rvs(mu = rl_mu, sigma = rl_sd, T = xi_obs)
                n_rl += 1
            else: # Gaussian noise
                y[jj,ii] = rng.normal(y0, np.sqrt(r0))

    return t_obs, y, R
    

def get_ln_rl_var(ln_vars,rl_vars,n_t,ii,SV):
    """
    Get the correct distribution to be used for a certain time step

    #### Input
    - `ln_vars`   ->  State variables that should be treated lognormally, 
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
    - `rl_vars`   ->  State variables that should be treated reverse lognormally, same as ln_vars
    - `n_t`       ->  Length of time vector to be compared to in the case ln_vars and rl_vars
                    are a list of indices
    - `ii`        ->  Index to be used in the case ln_vars and rl_vars are a list of indices
    - `SV`        ->  State variables to be used in the case ln_vars and rl_vars are callable

    #### Output
    - `ln_var`    ->  State variables that should be treated lognormally, 
                    as a list of indices between 0 and n_SV-1
    - `rl_var`    ->  State variables that should be treated reverse lognormally, 
                    as a list of indices between 0 and n_SV-1
    """

    # Get lognormally distributed state variables for this time step
    if callable(ln_vars):
        ln_var = ln_vars(SV)
    elif len(ln_vars) == n_t:
        ln_var = ln_vars[ii]
    else: # Same treatment for all time steps
        ln_var = ln_vars

    # Get reverse lognormally distributed state variables for this time step
    if callable(rl_vars):
        rl_var = rl_vars(SV)
    elif len(rl_vars) == n_t:
        rl_var = rl_vars[ii]
    else: # Same treatment for all time steps
        rl_var = rl_vars

    # Check if there is any overlap in lognormal and reverse lognormal variables
    s_ln_var = set(ln_var)
    if any(x in s_ln_var for x in rl_var):
        raise RuntimeError("State variables can not be lognormal and reverse lognormal at the same time!")

    return ln_var, rl_var


def transform_vars(x_a, x_b, ln_var, rl_var, xi, ensembles = False):
    """
    Transform state variables to mixed representation

    Two arrays of state variables are transformed to a mixed representation at the same time. 
    The method tests if a logarithm of a negative value is ever taken,
    and if so removes the index of the state variable for which this is the case from ln_var or rl_var

    #### Input
    - `x_a`       ->  State variables that need to be transformed, array of size n_SV
    - `x_b`       ->  State variables that need to be transformed, either an array of size n_SV
                    or an array of size n_SV x n_e (if ensembles == True)
    - `ln_var`    ->  State variables that should be treated lognormally, 
                    as a list of indices between 0 and n_SV-1
    - `rl_var`    ->  State variables that should be treated reverse lognormally, 
                    as a list of indices between 0 and n_SV-1
    - `xi`        ->  Parameter for the reverse lognormal distribution
    - `ensembles` ->  Boolean deciding if x_b contains multiple ensemble members 
                    If True, x_b is an array of size n_SV x n_e
                    Default is False

    #### Output
    - `x_a_mix`   ->  Mixed representation of x_a, such that 
                        x_a_mix[gs_var] = x_a
                        x_a_mix[ln_var] = log(x_a)
                        x_a_mix[rl_var] = log(xi - x_a)
    - `x_b_mix`   ->  Mixed representation of x_b, such that 
                        x_b_mix[gs_var] = x_b
                        x_b_mix[ln_var] = log(x_b)
                        x_b_mix[rl_var] = log(xi - x_b)
    - `ln_var`    ->  State variables that should be treated lognormally, 
                    as a list of indices between 0 and n_SV-1.
                    If a lognormal state variable had a negative value, this state variable
                    is instead treated as Gaussian, and the index is removed from ln_var
    - `rl_var`    ->  State variables that should be treated reverse lognormally, 
                    as a list of indices between 0 and n_SV-1.
                    If a reverse lognormal state variable had a value larger than xi, this state variable
                    is instead treated as Gaussian, and the index is removed from rl_var
    """

    # Initialize mixed variables
    x_a_mix = copy.copy(x_a)
    x_b_mix = copy.copy(x_b)

    ln_var_new = copy.copy(ln_var)
    rl_var_new = copy.copy(rl_var)

    if ensembles: # x_b is an array of size n_SV x n_e
        # Lognormal variables
        try:
            x_a_mix[ln_var] = np.log(x_a[ln_var])
            x_b_mix[ln_var,:] = np.log(x_b[ln_var,:])
        except RuntimeWarning:
            # Value of negative value taken, find component where this is the case and
            # use Gaussian distribution instead for this time step
            print('Warning: logarithm of negative value taken')
            for iLN in range(len(ln_var)):
                try:
                    x_a_mix[ln_var[iLN]] = np.log(x_a[ln_var[iLN]])
                    x_b_mix[ln_var[iLN],:] = np.log(x_b[ln_var[iLN],:])
                except RuntimeWarning:
                    x_a_mix[ln_var[iLN]] = x_a[ln_var[iLN]]
                    x_b_mix[ln_var[iLN],:] = x_b[ln_var[iLN],:]
                    ln_var_new.remove(ln_var[iLN])

        # Reverse lognormal variables
        try:
            x_a_mix[rl_var] = np.log(xi - x_a[rl_var])
            x_b_mix[rl_var,:] = np.log(xi - x_b[rl_var,:])
        except RuntimeWarning:
            # Value of negative value taken, find component where this is the case and
            # use Gaussian distribution instead for this time step
            print('Warning: logarithm of negative value taken (rl)')
            for iRL in range(len(rl_var)):
                try:
                    x_a_mix[rl_var[iRL]] = np.log(xi - x_a[rl_var[iRL]])
                    x_b_mix[rl_var[iRL],:] = np.log(xi - x_b[rl_var[iRL],:])
                except RuntimeWarning:
                    x_a_mix[rl_var[iRL]] = x_a[rl_var[iRL]]
                    x_b_mix[rl_var[iRL],:] = x_b[rl_var[iRL],:]
                    rl_var_new.remove(rl_var[iRL])
    else: # x_b is an array of size n_SV
        # Lognormal variables
        try:
            x_a_mix[ln_var] = np.log(x_a[ln_var])
            x_b_mix[ln_var] = np.log(x_b[ln_var])
        except RuntimeWarning:
            # Value of negative value taken, find component where this is the case and
            # use Gaussian distribution instead for this time step
            print('Warning: logarithm of negative value taken')
            for iLN in range(len(ln_var)):
                try:
                    x_a_mix[ln_var[iLN]] = np.log(x_a[ln_var[iLN]])
                    x_b_mix[ln_var[iLN]] = np.log(x_b[ln_var[iLN]])
                except RuntimeWarning:
                    x_a_mix[ln_var[iLN]] = x_a[ln_var[iLN]]
                    x_b_mix[ln_var[iLN]] = x_b[ln_var[iLN]]
                    ln_var_new.remove(ln_var[iLN])

        # Reverse lognormal variables
        try:
            x_a_mix[rl_var] = np.log(xi - x_a[rl_var])
            x_b_mix[rl_var] = np.log(xi - x_b[rl_var])
        except RuntimeWarning:
            # Value of negative value taken, find component where this is the case and
            # use Gaussian distribution instead for this time step
            print('Warning: logarithm of negative value taken (rl)')
            for iRL in range(len(rl_var)):
                try:
                    x_a_mix[rl_var[iRL]] = np.log(xi - x_a[rl_var[iRL]])
                    x_b_mix[rl_var[iRL]] = np.log(xi - x_b[rl_var[iRL]])
                except RuntimeWarning:
                    x_a_mix[rl_var[iRL]] = x_a[rl_var[iRL]]
                    x_b_mix[rl_var[iRL]] = x_b[rl_var[iRL]]
                    rl_var_new.remove(rl_var[iRL])

    return x_a_mix, x_b_mix, ln_var_new, rl_var_new


def rmse(SV_true,SV_DA,period_DA = 1):
    """
    Calculate the root mean square error of the analysis with respect to the true state

    #### Input
      - `SV_true`   ->  State variables of truth, array of size n_SV x n_t,
                        with n_SV the number of variables, and n_t the number of time steps
      - `SV_DA`     ->  State variables of analysis from DA, array of size n_SV x n_t_DA,
                        with n_SV the number of variables, and n_t_a the number of time steps
    - `period_DA`   ->  Analysis period, in steps of true time, such that
                            n_t = n_t_DA * period_DA
                        Default = 1, such that n_t = n_t_DA

    #### Output
      - `RMSE`      ->  Root mean square error of all state variables over the entire time period
    """

    return np.sqrt(np.nanmean((SV_true[:,::period_DA] - SV_DA)**2))
    
def rmse_time(SV_true, SV_DA, period_DA = 1, ln_vars = [], rl_vars = [], xi = 0.0):
    """
    Calculate the root mean square error of the analysis with respect to the true state

    #### Input
    - `SV_true`   ->  State variables of truth, array of size n_SV x n_t,
                    with n_SV the number of variables, and n_t the number of time steps
    - `SV_DA`     ->  State variables of analysis from DA, array of size n_SV x n_t_DA,
                    with n_SV the number of variables, and n_t_a the number of time steps
    - `period_DA` ->  Analysis period, in steps of true time, such that
                        n_t = n_t_DA * period_DA
                    Default = 1, such that n_t = n_t_DA
    - `ln_vars`   ->  State variables that should be treated lognormally, 
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
    - `rl_vars`   ->  State variables that should be treated reverse lognormally, same as ln_vars
    - `xi`        ->  Parameter for the reverse lognormal distribution

    #### Output
    - `RMSE`      ->  Root mean square error of all state variables over the entire time period
    """

    X = copy.copy(SV_true[:,::period_DA])
    Y = copy.copy(SV_DA)
    n_t_DA = SV_DA[0,:].size

    for ii in range(1,n_t_DA):  # Initial value (y[:,0]) has no observation and no noise

        # Get indices for lognormally and reverse lognormally distributed state variables 
        # and observations for this time step
        ln_var,  rl_var  = get_ln_rl_var(ln_vars, rl_vars, n_t_DA, ii, X[:,ii])
        
        # Transform values
        try:
            X[ln_var,ii], Y[ln_var,ii] = np.log(X[ln_var,ii]), np.log(Y[ln_var,ii])
        except RuntimeWarning:
            pass
        
        try:
            X[rl_var,ii], Y[rl_var,ii] = np.log(xi - X[rl_var,ii]), np.log(xi - Y[rl_var,ii])
        except RuntimeWarning:
            pass

    r = np.sqrt(np.mean((X - Y)**2, axis = 0))
    
    return np.nanmean(r)




# Timeout exception
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)