#%%
# Predicting db/dt for the two regimes separately
import numpy as np
import pysr
import pandas as pd

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')

from utils import *

# load the fields from the snapshots h5 files
my_fields = [read_snapshots('RB_snaps/snapshots_2e6_1/snapshots_s'+str(i)+'.h5')
             for i in range(1, 5)]

def create_array(x, index, datatype=np.float32):
    '''
    Create an array for the quantity x from the index_th key in my_files
    dictionary and set its dataype to datatype.
    '''
    return np.concatenate([my_fields[i][index][x] for i in range(len(my_fields))],
                                                        axis=0).astype(datatype)
buoyancy = create_array('buoyancy', 0)
div_grad_b = create_array('div_grad_b', 1)
lift_tau_b2 = create_array('lift_tau_b2', 7)

grad_b = create_array('grad_b', 4)
velocity = create_array('velocity', 10)

db_dt = derivative_wrt_time(buoyancy, 0.25)
db_dt = db_dt.reshape(200, 128, 2, 32, 2).mean(axis=(2,4))

# perform mean-pooling so that arrays become (25, 64, 32) instead of (200, 256, 64)
buoyancy = buoyancy.reshape(200, 128, 2, 32, 2).mean(axis=(2,4))
div_grad_b = div_grad_b.reshape(200, 128, 2, 32, 2).mean(axis=(2,4))
lift_tau_b2 = lift_tau_b2.reshape(200, 128, 2, 32, 2).mean(axis=(2,4))

grad_b_x = grad_b[:, 0, :, :]
grad_b_z = grad_b[:, 1, :, :]
grad_b_x = grad_b_x.reshape(200, 128, 2, 32, 2).mean(axis=(2,4))
grad_b_z = grad_b_z.reshape(200, 128, 2, 32, 2).mean(axis=(2,4))

ux = velocity[:, 0, :, :]
uz = velocity[:, 1, :, :]
ux = ux.reshape(200, 128, 2, 32, 2).mean(axis=(2,4))
uz = uz.reshape(200, 128, 2, 32, 2).mean(axis=(2,4))

# Now we need to convert the data to a format that PySR can understand
# We need an X array and a y array.
# We will take the reshaped div_grad_u_x and div_grad_u_z as the y array
# and everything else as the X array.
u_grad_b = ux * grad_b_x + uz * grad_b_z
#%%
X1 = np.concatenate([arr[:45, :, :].reshape(-1,1) for arr in [
    div_grad_b, lift_tau_b2, u_grad_b
    ]], axis=1)
y1 = db_dt[:45, :, :].reshape(-1,1)
print(X1.shape, y1.shape)

# Delete unnecessary variables
# del buoyancy, div_grad_b, lift_tau_b2, grad_b, velocity, db_dt, grad_b_x, grad_b_z,
# del u_grad_b, ux, uz, my_fields

# perform multiple linear regression on the data

model = sm.OLS(y1, X1)
results = model.fit()
print(results.summary())

# compute vif values
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
vif["features"] = ['div_grad_b', 'lift_tau_b2', 'u_grad_b']
print(vif)
#%%
model = pysr.PySRRegressor(binary_operators=["+", "*", "-"],
                    # unary_operators=["exp", "abs", "relu",
                    #                     "sqrt", "square", "cube",
                    #                     "log10", "log2", "log",
                    #                     "sin", "cos", "tan",
                    #                     "atan", "sinh", "cosh", "tanh",
                    #                     "sign", "floor", "ceil"],
                    # use_frequency=False,
                    # use_frequency_in_tournament=False,
                    # adaptive_parsimony_scaling=1,
                    verbosity=0)

model.fit(X1, y1)
print("R^2 = ", model.score(X1, y1))
print(model.sympy())

plotting(model, X1, y1, 0, 'db_dt in the first regime')
#%%
# plot collinearity matrix
corr = np.corrcoef(X1, rowvar=False)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap='viridis')
plt.show()
#%%
X2 = np.concatenate(
    [arr[45:, :, :].reshape(-1,1) for arr in 
     [div_grad_b, lift_tau_b2, u_grad_b]], 
     axis=1
     ).astype(np.float32)
y2 = db_dt[45:, :, :].reshape(-1,1).astype(np.float32)

model.fit(X2, y2)
print("R^2 = ", model.score(X2, y2))
print(model.sympy())

plotting(model, X2, y2, 0, 'db_dt in the second regime')

model_ols_1 = sm.OLS(y1, X1)
results_1 = model_ols_1.fit()
print(results_1.summary())

model_ols_2 = sm.OLS(y2, X2)
results_2 = model_ols_2.fit()
print(results_2.summary())
# %%
