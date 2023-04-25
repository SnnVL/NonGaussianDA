"""
Functions needed for solving the different Lorenz models

#### Methods:
- `sol_L63`         ->  Solution to the Lorenz-63 model
- `sol_cL63`        ->  Solution to the coupled Lorenz-63 model
- `sol_gL63`        ->  Solution to the generalized Lorenz-63 model
- `sol_L96`         ->  Solution to the Lorenz-96 model
- `sol_L05`         ->  Solution to the Lorenz-05 model 2

#### Author:
Senne Van Loon
Cooperative Institute for Research in the Atmosphere (CIRA),
Colorado State University,
3925A West Laporte Ave, Fort Collins, CO 80521

#### References and acknowledgements:
* Lorenz, E. N. (1963). Deterministic nonperiodic flow. Journal of atmospheric sciences, 20(2), 130-141.
* Lorenz, E. N. (1996, September). Predictability: A problem partly solved. In Proc. Seminar on predictability (Vol. 1, No. 1).
* Lorenz, E. N. (2005). Designing chaotic models. Journal of the atmospheric sciences, 62(5), 1574-1587.
* Shen, B. W. (2014). Nonlinear feedback in a five-dimensional Lorenz model. Journal of the Atmospheric Sciences, 71(5), 1701-1723.
* Shen, B. W. (2019). Aggregated negative feedback in a generalized Lorenz model. International Journal of Bifurcation and Chaos, 29(03), 1950037.
"""

import numpy as np
from ctypes import c_double, c_int, CDLL

# Load C-library
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
c_mod = CDLL(dir_path+"/C_Lorenz.so")

def sol_L63(t, x0, p = np.array([10.0,28.0,8.0/3.0])):
    N = 3
    if x0.size != N:
        raise RuntimeError("Incorrect initial conditions!")

    t0 = c_double(t[0])
    dt = c_double(t[1]-t[0])
    nt = t.size

    x0_c = (c_double * N)(*x0)
    sol_c = (c_double * (N*nt))()

    s = c_double(p[0])
    r = c_double(p[1])
    b = c_double(p[2])

    c_mod.sol_L63_(t0,dt,c_int(nt),x0_c,sol_c,s,r,b)

    return np.reshape(np.array(sol_c[:]),(N,nt),order='F')

def sol_cL63(t, x0, p = np.array([10.0,28.0,8.0/3.0]), c = np.array([0.0,0.0,0.0]), n=4):
    N = 3*n
    if x0.size != N:
        raise RuntimeError("Incorrect initial conditions!")

    t0 = c_double(t[0])
    dt = c_double(t[1]-t[0])
    nt = t.size

    x0_c = (c_double * N)(*x0)
    sol_c = (c_double * (N*nt))()

    s = c_double(p[0])
    r = c_double(p[1])
    b = c_double(p[2])
    cx = c_double(c[0])
    cy = c_double(c[1])
    cz = c_double(c[2])

    c_mod.sol_cL63_(t0,dt,c_int(nt),x0_c,sol_c,c_int(n),s,r,b,cx,cy,cz)

    return np.reshape(np.array(sol_c[:]),(N,nt),order='F')

def sol_gL63(t, x0, M = 5, p = np.array([10.0,28.0,0.5])):
    if x0.size != M:
        raise RuntimeError("Incorrect initial conditions!")
    if np.mod(M,2) == 0:
        raise RuntimeError("The dimension M must be odd!")

    t0 = c_double(t[0])
    dt = c_double(t[1]-t[0])
    nt = t.size

    x0_c = (c_double * M)(*x0)
    sol_c = (c_double * (M*nt))()

    s = c_double(p[0])
    r = c_double(p[1])
    a2 = c_double(p[2])

    c_mod.sol_gL63_(t0,dt,c_int(nt),x0_c,sol_c,c_int(M),s,r,a2)

    return np.reshape(np.array(sol_c[:]),(M,nt),order='F')

def sol_L96(t, x0, N, F):

    if x0.size != N:
        raise RuntimeError("Incorrect initial conditions!")

    t0 = c_double(t[0])
    dt = c_double(t[1]-t[0])
    nt = t.size

    x0_c = (c_double * N)(*x0)
    sol_c = (c_double * (N*nt))()

    c_mod.sol_L96_(t0,dt,c_int(nt),x0_c,sol_c,c_int(N),c_double(F))

    return np.reshape(np.array(sol_c[:]),(N,nt),order='F')

def sol_L05(t, x0, N, F, K):

    if x0.size != N:
        raise RuntimeError("Incorrect initial conditions!")

    t0 = c_double(t[0])
    dt = c_double(t[1]-t[0])
    nt = t.size

    x0_c = (c_double * N)(*x0)
    sol_c = (c_double * (N*nt))()

    c_mod.sol_L05_(t0,dt,c_int(nt),x0_c,sol_c,c_int(N),c_double(F),c_int(K))

    return np.reshape(np.array(sol_c[:]),(N,nt),order='F')