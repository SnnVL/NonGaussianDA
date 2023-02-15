""" 
Run a nongaussian twin experiment with Lorenz-63, for different 3D methods (direct assimilation of observations).
"""

# Basic modules
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import time

# Lorenz-63 and DA methods
import sys
sys.path.append("../Models/")
sys.path.append("../DataAssimilation/")
import mod_L63 as L63
import mod_variationalDA as var
import mod_KalmanDA as kf
import mod_MLEF as mlef
import mod_DA_general as da


# Options for DA run
da_method = '3DVAR'                            # DA method, either '3DVAR', 'KF', or 'MLEF'
n_runs = 50                                 # Number of DA runs
n_wind = 250                                # Number of DA windows for each run
obs_vars = 'xyz'                            # Observed variables, either 'xy', or 'xyz'
n_e = 3                                     # Number of ensemble members (only for MLEF)
ml_method = 'array'                         # Nongaussian decision function, either  
                                            #   'array',    selection based on observations before DA run
                                            #   'function', selection from decision function every DA window
w = 12
s_cutoff = 1.2
seed = None                                 # Seed for random number generator, can be set to None
SV_init = [-5.0,-6.0,22.0]                  # Initial values of the Lorenz-63 run
descriptor = 'median'                       # Descriptor to optimize 3DVAR, 
                                            # either 'mode', 'mean', or 'median'


###########################################################################################
###########################################################################################
###########################################################################################

def one_run(jj):
    global da_method
    global n_wind
    global period_obs
    global obs_vars
    global var_obs
    global n_e
    global ml_method
    global ml_file
    global seed
    global SV_init
    global descriptor

    # Load the machine learning model
    with open(ml_file,'rb') as f:
        clf = pickle.load(f)
        scaler = pickle.load(f)
        info = pickle.load(f)
        p = info['p']               # Parameters of L63, typically (10,28,8/3)
        dt = info['dt']             # Time step of L63
    rng = np.random.default_rng(seed)
    SV_init += rng.standard_normal(3)

    # Nature run
    t_max = n_wind*period_obs*dt        # End time of the Lorenz-63 run
    t_eval = np.arange(0.0, t_max, dt)  # Evaluation time
    t, SV = L63.sol_l63([0.0, t_max], SV_init, p, t_eval, 'RK45')
    xi_SV = np.nanmax(SV[2,:])+5.0      # Parameter of the reverse lognormal distribution


    ###########################################################################################
    ###########################################################################################
    ###########################################################################################


    # Create observation operator
    n_SV = 3
    if da_method == 'MLEF':
        if obs_vars == 'xyz':
            def obs_h(x):
                return x
            n_obs = 3
        if obs_vars == 'xy':
            def obs_h(x):
                return x[:2]
            n_obs = 2
    elif (da_method == '3DVAR') or (da_method == 'KF'):
        if obs_vars == 'xyz':
            n_obs = 3
            obs_h = np.eye(n_SV)
        if obs_vars == 'xy':
            n_obs = 2
            obs_h = np.zeros((n_obs,n_SV))
            obs_h[0,0] = 1.0
            obs_h[1,1] = 1.0
    else:
        raise RuntimeError("DA method not recognized.")

    # Predict distribution from which to sample z-variable
    if obs_vars == 'xyz':
        # Predict distribution from nature run
        X_data = scaler.transform(np.transpose(SV[:2,:]))
        z_pred = clf.predict(X_data)
        # Define observation times when lognormal or reverse lognormal noise should be added
        ln_vars_ML = []
        rl_vars_ML = []
        for ii in range(t.size):
            if ii % period_obs == 0:
                if z_pred[ii] > 0.5: # Lognormal distribution predicted
                    ln_vars_ML.append([2])
                    rl_vars_ML.append([])
                elif z_pred[ii] < -0.5: # Reverse lognormal distribution predicted
                    ln_vars_ML.append([])
                    rl_vars_ML.append([2])
                else: # Gaussian distribution predicted
                    ln_vars_ML.append([])
                    rl_vars_ML.append([])
    if obs_vars == 'xy':
        # If z is not observed, ML model not necessary
        ln_vars_ML = []
        rl_vars_ML = []

    # Generate observations
    t_obs, y, R = da.gen_obs(t, SV, period_obs, obs_h, var_obs, \
        ln_vars = ln_vars_ML, rl_vars = rl_vars_ML, xi_obs = xi_SV, seed = seed)

    # Create initial background error covariance matrix
    if da_method == 'MLEF':
        rng = np.random.default_rng(seed)
        sqrtP_a = rng.normal(0,1.0, size = (n_SV,n_e))
    elif (da_method == '3DVAR') or (da_method == 'KF'):
        B = L63.create_B_init()
    else:
        raise RuntimeError("DA method not recognized.")


    ###########################################################################################
    ###########################################################################################
    ###########################################################################################


    # DA initial guess
    init_guess = SV_init + rng.standard_normal(3)

    # Model function, of the form x = model(t,x_0)
    def model_l63(t, x0):
        _, y = L63.sol_l63([t[0],t[-1]], x0, p, t, meth='RK45')
        return y

    # Decision function for lognormal state variables (for a single prediction)
    def f_ln_vars(SV):
        X = scaler.transform(SV[:2].reshape(1,-1))
        z_pred = clf.predict(X)
        if z_pred > 0.5:
            return [2]
        else:
            return []
    # Decision function for reverse-lognormal state variables (for a single prediction)
    def f_rl_vars(SV):
        X = scaler.transform(SV[:2].reshape(1,-1))
        z_pred = clf.predict(X)
        if z_pred < -0.5:
            return [2]
        else:
            return []

    # Create decision functions/arrays
    if ml_method == 'array':
        ln_vars = []
        rl_vars = []
        for ii in range(t_obs.size):
            ln_vars.append(f_ln_vars(y[:2,ii]))
            rl_vars.append(f_rl_vars(y[:2,ii]))
    if ml_method == 'function':
        ln_vars = f_ln_vars
        rl_vars = f_rl_vars

    # Store different decision functions for different (non)gaussian methods
    n_meths = 7
    meths = [ \
        'Gaussian ', \
        'Lognormal', \
        'Rev logn ', \
        'G-LogNorm', \
        'G-RevLog ', \
        'All mixed', \
        'noDA     ' \
    ]

    ln_vars_SV_OBJ = np.array([ \
        [],      # Gaussian for all t
        [2],     # Lognormal for all t
        [],      # Reverse Lognormal for all t
        ln_vars, # Gaussian - Lognormal ML
        [],      # Gaussian - Reverse Lognormal ML
        ln_vars, # Gaussian - Lognormal - Reverse Lognormal ML
        [],      # No DA (background run)
    ], dtype = object)
    rl_vars_SV_OBJ = np.array([ \
        [],      # Gaussian for all t
        [],      # Lognormal for all t
        [2],     # Reverse Lognormal for all t
        [],      # Gaussian - Lognormal ML
        rl_vars, # Gaussian - Reverse Lognormal ML
        rl_vars, # Gaussian - Lognormal - Reverse Lognormal ML
        [],      # No DA (background run)
    ], dtype = object)
    if obs_vars == 'xyz':
        ln_vars_obs_OBJ = ln_vars_SV_OBJ
        rl_vars_obs_OBJ = rl_vars_SV_OBJ
    elif obs_vars == 'xy':
        ln_vars_obs_OBJ = np.array([[],[],[],[],[],[],[-1.0]], dtype = object)
        rl_vars_obs_OBJ = np.array([[],[],[],[],[],[],[-1.0]], dtype = object)

    # Initialize analysis and background state variables
    n_t_obs = t_obs.size           # number of observations
    n_t = n_t_obs * period_obs + 1 # number of total time steps
    X_A = np.empty((n_meths, n_SV, n_t_obs))
    X_B = np.empty((n_meths, n_SV, n_t))

    for iM in range(n_meths-1):
        t1 = time.time()
        if da_method == '3DVAR':

            varmeth = 'min'             # Method for minimizing, either 'min', or 'root'
            if descriptor == 'mode':
                l_SV, l_obs = 1.0, 1.0
            elif descriptor == 'mean':
                l_SV, l_obs =-0.5,-0.5
            elif descriptor == 'median':
                l_SV, l_obs = 0.0, 0.0

            X_A[iM,:,:], X_B[iM,:,:], _ = var.var3d( \
                init_guess, t_obs, period_obs, y, obs_h, B, R, \
                model_l63, \
                ln_vars_SV = ln_vars_SV_OBJ[iM], ln_vars_obs = ln_vars_obs_OBJ[iM], \
                rl_vars_SV = rl_vars_SV_OBJ[iM], rl_vars_obs = rl_vars_obs_OBJ[iM], \
                xi_SV = xi_SV, xi_obs = xi_SV, \
                method = varmeth, \
                l_SV = l_SV, l_obs = l_obs \
            )

        elif da_method == 'KF':
            
            # Model error covariance matrix
            Q = np.array([ \
                [0.1491, 0.1505, 0.0007], \
                [0.1505, 0.9048, 0.0014], \
                [0.0007, 0.0014, 0.9180] \
            ])

            X_A[iM,:,:], X_B[iM,:,:], _ = kf.kalman_filter( \
                init_guess, t_obs, period_obs, y, obs_h, B, R, Q, \
                model_l63, \
                ln_vars_SV = ln_vars_SV_OBJ[iM], ln_vars_obs = ln_vars_obs_OBJ[iM], \
                rl_vars_SV = rl_vars_SV_OBJ[iM], rl_vars_obs = rl_vars_obs_OBJ[iM], \
                xi_SV = xi_SV, xi_obs = xi_SV \
            )

        elif da_method == 'MLEF':
            
            X_A[iM,:,:], X_B[iM,:,:], _ = mlef.MLEF( \
                init_guess, t_obs, period_obs, n_e, y, obs_h, sqrtP_a, R, \
                model_l63, \
                ln_vars_SV = ln_vars_SV_OBJ[iM], ln_vars_obs = ln_vars_obs_OBJ[iM], \
                rl_vars_SV = rl_vars_SV_OBJ[iM], rl_vars_obs = rl_vars_obs_OBJ[iM], \
                xi_SV = xi_SV, xi_obs = xi_SV \
            )

        else:
            raise RuntimeError("DA method not recognized.")
        t2 = time.time()
        # print("Finished "+meths[iM]+" "+da_method+" run "+str(jj)+", t = "+str(t2-t1))
        
    # Background run
    iM += 1
    dt_obs = t_obs[1] - t_obs[0] # Assuming evenly spaced observations
    t_true = np.linspace(t_obs[0],t_obs[-1] + dt_obs, n_t) # Assuming one prediction window
    X_B[iM,:,:] = model_l63(t_true, init_guess)
    X_A[iM,:,:] = X_B[iM,:,:-1:period_obs]

    RMSE_A = np.empty(n_meths)
    RMSE_B = np.empty(n_meths)
    # print("Root mean square errors for "+da_method)
    for iM in range(n_meths):
        RMSE_A[iM] = da.rmse(SV, X_A[iM,:,:], period_DA=period_obs)
        RMSE_B[iM] = da.rmse(SV, X_B[iM,:,:-1])

        # print(meths[iM]+ \
        #     ': r_a = '+format(np.round(RMSE_A[iM],3),".3f")+ \
        #     ', r_b = '+format(np.round(RMSE_B[iM],3),".3f"))

    # print("")
    MAE_A = np.empty(n_meths)
    MAE_B = np.empty(n_meths)
    # print("Mean absolute errors for "+da_method)
    for iM in range(n_meths):
        MAE_A[iM] = da.mae(SV, X_A[iM,:,:], period_DA=period_obs)
        MAE_B[iM] = da.mae(SV, X_B[iM,:,:-1])

        # print(meths[iM]+ \
        #     ': r_a = '+format(np.round(MAE_A[iM],3),".3f")+ \
        #     ', r_b = '+format(np.round(MAE_B[iM],3),".3f"))

    # print("")
    MeAE_A = np.empty(n_meths)
    MeAE_B = np.empty(n_meths)
    # print("Median absolute errors for "+da_method)
    for iM in range(n_meths):
        MeAE_A[iM] = da.medae(SV, X_A[iM,:,:], period_DA=period_obs)
        MeAE_B[iM] = da.medae(SV, X_B[iM,:,:-1])

        # print(meths[iM]+ \
        #     ': r_a = '+format(np.round(MeAE_A[iM],3),".3f")+ \
        #     ', r_b = '+format(np.round(MeAE_B[iM],3),".3f"))

    return RMSE_A, RMSE_B, MAE_A, MAE_B, MeAE_A, MeAE_B

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################


from multiprocessing import Pool

if __name__ == '__main__':

    n_meths = 7
    ml_file_str =  'w'+str(w)+'_s'+str(np.round(s_cutoff,2))
    ml_file = './data/kNN_l63_'+ml_file_str+'.pkl'    # Location of the trained ML model
    po_vec = np.arange(20,220,20)
    vo_vec = np.arange(0.5,5.0,0.5)  

    RMSE_A = np.empty((po_vec.size,vo_vec.size,n_runs,n_meths))
    RMSE_B = np.empty((po_vec.size,vo_vec.size,n_runs,n_meths))
    MAE_A = np.empty((po_vec.size,vo_vec.size,n_runs,n_meths))
    MAE_B = np.empty((po_vec.size,vo_vec.size,n_runs,n_meths))
    MeAE_A = np.empty((po_vec.size,vo_vec.size,n_runs,n_meths))
    MeAE_B = np.empty((po_vec.size,vo_vec.size,n_runs,n_meths))

    for ii, period_obs in enumerate(po_vec):
        for jj, var_obs in enumerate(vo_vec):

            with Pool(np.min([n_runs,5])) as p:
                X = p.map(one_run, range(n_runs))

            for kk in range(n_runs):
                RMSE_A[ii,jj,kk,:], RMSE_B[ii,jj,kk,:], \
                MAE_A[ii,jj,kk,:], MAE_B[ii,jj,kk,:], \
                MeAE_A[ii,jj,kk,:], MeAE_B[ii,jj,kk,:] = X[kk]

            print("Finished p = "+str(period_obs)+", s = "+str(var_obs))

    info = {
        "da_method": da_method,
        "n_runs": n_runs,
        "n_wind": n_wind,
        "period_obs": po_vec,
        "obs_vars": obs_vars,
        "var_obs": vo_vec,
        "n_e": n_e,
        "ml_method": ml_method,
        "seed": seed,
        "SV_init": SV_init,
        "w": w,
        "s": s_cutoff
    }
    with open('./data/ngDA_L63_' \
        +da_method+"_" \
        +obs_vars+"_" \
        +ml_file_str+"_" \
        +descriptor \
        +'.pkl','wb') as f:
        pickle.dump(RMSE_A, f)
        pickle.dump(RMSE_B, f)
        pickle.dump(MAE_A, f)
        pickle.dump(MAE_B, f)
        pickle.dump(MeAE_A, f)
        pickle.dump(MeAE_B, f)
        pickle.dump(info, f)
    print("Finished")