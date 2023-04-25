""" 
Train machine learning model to detect the underlying distribution of the z-variable 
in the coupled Lorenz-63 model.
"""

# Basic modules
import numpy as np 
from scipy import stats
import pickle
import time

# Lorenz-63
import sys
sys.path.append("../Models/")
# import mod_coupledLorenz as L63
from LorenzModels import sol_cL63

# ML modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Window for calculating the skewness
w = 12
s_cutoff = -stats.norm.ppf(0.1)

# Add random noise to initial conditions
seed = 42
rng = np.random.default_rng(seed)

# Initial conditions
n = 10
SV_init_0 = np.array([-5.0,-5.0,25.0])
SV_init = np.empty(3*n)
for ii in range(n):
    SV_init[3*ii:3*ii+3] = SV_init_0 + rng.normal(loc = 0.0, scale = 3.0, size = (3))

# Generate data
c_all = 0.1
p = np.array([10.0,28.0,8.0/3.0])   # Parameters of Lorenz-63
c = np.array([c_all,c_all,c_all])         # Parameters of Lorenz-63
t_span = [0.0,1000.0]                # Time span
dt = 0.01                           # Time step
t_eval = np.arange(t_span[0],t_span[1],dt)

# Solve Lorenz model
SV = sol_cL63(t_eval,SV_init,p=p,c=c,n=n)

# Calculate z-score of skewtest
z = SV[2,:]
sp = np.zeros((z.size,2))
sp[w:z.size-w,:]=[stats.skewtest(z[iz-w:iz+w]) for iz in range(w,z.size-w)]
s = sp[:,0]

# Create feature vector (x and y variables) and output (skewness)
X_data = np.transpose(SV[:2,:])
y_data = np.zeros_like(s)
# 0 when Gaussian, 1 when lognormal, -1 when reverse lognormal
y_data[s >= s_cutoff] =  1 
y_data[s <=-s_cutoff] = -1 

# Remove data points for spinup
X_data = X_data[100:-50,:]
y_data = y_data[100:-50]

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)  
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  

# Create and train model
clf = KNeighborsClassifier(n_neighbors=15,weights='distance')
t1 = time.time()
clf.fit(X_train, y_train)
acc_score = clf.score(X_test,y_test)
t2 = time.time()
print('Training kNN model: t = ' \
    + str(t2-t1)+ ', accuracy = '+str(acc_score) \
)

info = {
    "SV_init": SV_init,
    "p": p,
    "c": c,
    "n": n,
    "w": w,
    "t_span": t_span,
    "dt": dt,
    "Classifier": "KNeighborsClassifier",
    "accuracy": acc_score,
    "s_cutoff":s_cutoff
}

# Save model and scaler to file
fName = 'kNN_cl63_n' + str(n) + '_c' + str(c_all) + '.pkl'
with open('./data/'+fName,'wb') as f:
    pickle.dump(clf, f)
    pickle.dump(scaler, f)
    pickle.dump(info, f)
