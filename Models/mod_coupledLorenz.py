""" 
Functions needed for solving the coupled Lorenz 63 model

#### Methods:
- `sol_l63_n`         ->  Solution to the coupled Lorenz-63 model

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


def l63_fun_n(t, SV, s, r, b, cx, cy, cz, n):
    """
    Equations of the coupled Lorenz 63 model, vectorized
    """

    if SV[:,0].size != 3*n:
        raise RuntimeError("Size error in Lorenz63")

    dSV_dt = np.empty_like(SV)
    for ii in range(n):
        x = SV[0+3*ii,:]
        y = SV[1+3*ii,:]
        z = SV[2+3*ii,:]

        dSV_dt[0+3*ii,:] = -s*(x-y) + cx * SV[np.mod(0+3*ii+3,3*n),:]
        dSV_dt[1+3*ii,:] = r*x-y-z*x+ cy * SV[np.mod(1+3*ii+3,3*n),:]
        dSV_dt[2+3*ii,:] = x*y-b*z  + cz * SV[np.mod(2+3*ii+3,3*n),:]

    return dSV_dt


def sol_l63_n(t_span, \
        SV_init, \
        p = np.array([10.0,28.0,8.0/3.0]), \
        c = np.array([0.0,0.0,0.0]), \
        n = 4, \
        t_eval = None, \
        meth = 'RK45'):
    """
    Solution to the coupled Lorenz 63 model 
    
    #### Input
    - `t_span`  ->  Beginning and endpoints in time
    - `SV_init` ->  Initial value for all state variables, array of size n_SV
    - `p`       ->  Parameters of the Lorenz model, (sigma, rho, beta)
                    Default = (10, 28, 8/3)
    - `c`       ->  Coupling parameters of the Lorenz model, (cx, cy, cz)
                    Default = (0, 0, 0)     (No coupling)
    - `n`       ->  Number of Lorenz63 systems (default = 4)
    - `t_eval`  ->  Time points where the model is evaluated
    - `meth`    ->  String indicating the integration method to use

    #### Output
    - `t`       ->  Time vector of size n_t
    - `y`       ->  State variables at every time point, array of size n_SV x n_t
    """

    sol = solve_ivp( \
        l63_fun_n, \
        t_span, \
        SV_init, \
        method = meth, \
        t_eval = t_eval, \
        vectorized = True, \
        args = (p[0],p[1],p[2],c[0],c[1],c[2],n) \
    )

    # Return none if integration was unsuccessful
    if not sol.success:
        return None, None 

    return sol.t, sol.y
