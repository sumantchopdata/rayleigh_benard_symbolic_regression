#%%
# Predicting du_x/dt for the two regimes separately
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
div_grad_u = create_array('div_grad_u', 2)
lift_tau_u2 = create_array('lift_tau_u2', 8)
ez = create_array('ez', 3)

grad_p = create_array('grad_p', 5)
velocity = create_array('velocity', 10)

grad_u = create_array('grad_u', 6)

du_dt = derivative_wrt_time(velocity, 0.25)

# perform mean-pooling so that arrays become (50, 128, 32) instead of (200, 256, 64)
buoyancy = buoyancy.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)

velocity_x = velocity[:, 0, :, :]
velocity_z = velocity[:, 1, :, :]

velocity_x = velocity_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)
velocity_z = velocity_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)
                                                                                            
div_grad_u_x = div_grad_u[:, 0, :, :]
div_grad_u_z = div_grad_u[:, 1, :, :]

lift_tau_u2_x = lift_tau_u2[:, 0, :, :].astype(np.float32)
lift_tau_u2_z = lift_tau_u2[:, 1, :, :].astype(np.float32)

lift_tau_u2_x = lift_tau_u2_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5))
lift_tau_u2_z = lift_tau_u2_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5))

div_grad_u_x = div_grad_u_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
div_grad_u_z = div_grad_u_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)

grad_p_x = grad_p[:, 0, :, :]
grad_p_z = grad_p[:, 1, :, :]
grad_p_x = grad_p_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
grad_p_z = grad_p_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)

grad_x_ux = grad_u[:, 0, 0, :, :]
grad_z_ux = grad_u[:, 0, 1, :, :]
grad_x_uz = grad_u[:, 1, 0, :, :]
grad_z_uz = grad_u[:, 1, 1, :, :]

grad_x_ux = grad_x_ux.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
grad_z_ux = grad_z_ux.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
grad_x_uz = grad_x_uz.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
grad_z_uz = grad_z_uz.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)

dux_dt = du_dt[:, 0, :, :]
duz_dt = du_dt[:, 1, :, :]

dux_dt = dux_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)
duz_dt = duz_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)

ez = ez.reshape(200,2).astype(np.float32)
ez = ez.reshape(50, 4, 2).mean(axis=1)
ez = np.tile(ez[:, np.newaxis], (1,128,16))

u_grad_u = np.multiply(velocity_x, grad_x_ux) + np.multiply(velocity_z, grad_z_ux)

X = np.concatenate(
    [arr[:45, :, :].reshape(-1,1) for arr in 
     [grad_p_x, div_grad_u_x, buoyancy, u_grad_u]], 
     axis=1
     ).astype(np.float32)

y = dux_dt[:45, :, :].reshape(-1,1).astype(np.float32)
print(X.shape, y.shape)
#%%
# delete unnecessary variables to free up memory
del buoyancy, div_grad_u, lift_tau_u2, grad_p, velocity, grad_u, du_dt, my_fields
del velocity_x, velocity_z, div_grad_u_x, div_grad_u_z, lift_tau_u2_x, lift_tau_u2_z
del grad_p_x, grad_p_z, grad_x_ux, grad_z_ux, grad_x_uz, grad_z_uz, dux_dt
del ez, duz_dt, u_grad_u
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

model.fit(X, y)
print("R^2 = ", model.score(X, y))
print(model.sympy())
#%%
# perform multiple linear regression on the data

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# compute vif values
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
vif["features"] = ['grad_p_x', 'div_grad_u_x', 'buoyancy*ez', 'u_grad_u', 'lift_tau_u2_x']
print(vif)
#%%
# plot collinearity matrix
corr = np.corrcoef(X, rowvar=False)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap='viridis')
plt.show()
#%%