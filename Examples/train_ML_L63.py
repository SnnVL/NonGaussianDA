""" 
Train machine learning model to detect the underlying distribution of the z-variable in the Lorenz-63 model.
"""

# Basic modules
import numpy as np 
from scipy import stats
import pickle
import time

# Lorenz-63
import sys
sys.path.append("../Models/")
import mod_L63 as L63

# ML modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Window for calculating the skewness
w = 12

# Generate data
SV_init = [-3.0,-3.0,20.0]          # Initial conditions
p = np.array([10.0,28.0,8.0/3.0])   # Parameters of Lorenz-63
t_span = [0.0,1000.0]                # Time span
dt = 0.01                           # Time step
t_eval = np.arange(t_span[0],t_span[1],dt)

# Solve Lorenz model
integrationMethod = 'RK45'
t, SV = L63.sol_l63(t_span,SV_init,p,t_eval,meth=integrationMethod)

# Calculate z-score of skewtest
z = SV[2,:]
sp = np.zeros((z.size,2))
sp[w:z.size-w,:]=[stats.skewtest(z[iz-w:iz+w]) for iz in range(w,z.size-w)]
s = sp[:,0]

# Create feature vector (x and y variables) and output (skewness)
X_data = np.transpose(SV[:2,:])
y_data = np.zeros_like(s)
# 0 when Gaussian, 1 when lognormal, -1 when reverse lognormal
y_data[s >= 1] =  1 
y_data[s <=-1] = -1 

# Remove data points for spinup
X_data = X_data[50:-50,:]
y_data = y_data[50:-50]

# Split data: 60% train, 20% cross-validation, 20% test
X_train, X_test_CV, y_train, y_test_CV = train_test_split(X_data, y_data, test_size=0.4, random_state=42)
X_CV, X_test, y_CV, y_test = train_test_split(X_test_CV, y_test_CV, test_size=0.5, random_state=42)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_CV = scaler.transform(X_CV)  
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
    "w": w,
    "t_span": t_span,
    "dt": dt,
    "Classifier": "KNeighborsClassifier",
    "integrationMethod": integrationMethod
}

# Save model and scaler to file
fName = 'kNN_l63_w' + str(w) + '.pkl'
with open('./data/'+fName,'wb') as f:
    pickle.dump(clf, f)
    pickle.dump(scaler, f)
    pickle.dump(info, f)
