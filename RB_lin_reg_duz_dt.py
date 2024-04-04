#%%
# Run a multiple linear regression on the data for the du_dt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the data

from utils import *

# load the fields from the snapshots h5 files
my_fields = [read_snapshots('RB_snaps/snapshots_2e6_1/snapshots_s'+str(i)+'.h5') for i in range(1, 5)]

def create_array(x, index, datatype=np.float32):
    '''
    Create an array for the quantity x from the index_th key in my_files
    dictionary and set its dataype to datatype.
    '''
    return np.concatenate([my_fields[i][index][x] for i in range(len(my_fields))],
                                                        axis=0).astype(datatype)

buoyancy = create_array('buoyancy', 0)
div_grad_b = create_array('div_grad_b', 1)
div_grad_u = create_array('div_grad_u', 2)
ez = create_array('ez', 3)
grad_b = create_array('grad_b', 4)
grad_p = create_array('grad_p', 5)
grad_u = create_array('grad_u', 6)
lift_tau_b2 = create_array('lift_tau_b2', 7)
lift_tau_u2 = create_array('lift_tau_u2', 8)
pressure = create_array('pressure', 9)
velocity = create_array('velocity', 10)
vorticity = create_array('vorticity', 11)

db_dt = derivative_wrt_time(buoyancy, 0.25)
du_dt = derivative_wrt_time(velocity, 0.25)
dvort_dt = derivative_wrt_time(vorticity, 0.25)
dp_dt = derivative_wrt_time(pressure, 0.25)

# perform mean-pooling so that arrays become (50, 128, 32) instead of (200, 256, 64)
# buoyancy = buoyancy.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)

velocity_x = velocity[:, 0, :, :]
velocity_z = velocity[:, 1, :, :]

# velocity_x = velocity_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)
# velocity_z = velocity_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)
                                                                                            
div_grad_u_x = div_grad_u[:, 0, :, :]
div_grad_u_z = div_grad_u[:, 1, :, :]

lift_tau_u2_x = lift_tau_u2[:, 0, :, :].astype(np.float32)
lift_tau_u2_z = lift_tau_u2[:, 1, :, :].astype(np.float32)

# lift_tau_u2_x = lift_tau_u2_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5))
# lift_tau_u2_z = lift_tau_u2_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5))

# div_grad_u_x = div_grad_u_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
# div_grad_u_z = div_grad_u_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)

grad_p_x = grad_p[:, 0, :, :]
grad_p_z = grad_p[:, 1, :, :]
# grad_p_x = grad_p_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
# grad_p_z = grad_p_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)

grad_x_ux = grad_u[:, 0, 0, :, :]
grad_z_ux = grad_u[:, 0, 1, :, :]
grad_x_uz = grad_u[:, 1, 0, :, :]
grad_z_uz = grad_u[:, 1, 1, :, :]

# grad_x_ux = grad_x_ux.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
# grad_z_ux = grad_z_ux.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
# grad_x_uz = grad_x_uz.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
# grad_z_uz = grad_z_uz.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)

dux_dt = du_dt[:, 0, :, :]
duz_dt = du_dt[:, 1, :, :]

# dux_dt = dux_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)
# duz_dt = duz_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)

# ez = ez.reshape(50,4,2).mean(axis=(1)).astype(np.float32)

# vorticity = vorticity.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)
# dvort_dt = dvort_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)

grad_b_x = grad_b[:, 0, :, :]
grad_b_z = grad_b[:, 1, :, :]

u_grad_u_x = np.multiply(velocity_x, grad_x_ux) + np.multiply(velocity_z, grad_z_ux)
u_grad_u_z = np.multiply(velocity_x, grad_x_uz) + np.multiply(velocity_z, grad_z_uz)

u_grad_b = velocity_x * grad_b_x + velocity_z * grad_b_z

# normalize the data to be between 0 and 1
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

for arr in [grad_p_z, div_grad_u_z, buoyancy, u_grad_u_z, vorticity, dvort_dt, duz_dt,
            grad_p_x, div_grad_u_x, u_grad_u_x, dux_dt, div_grad_b, lift_tau_b2,
            u_grad_b, lift_tau_u2_x, lift_tau_u2_z]:
    arr = normalize(arr)

X1 = np.concatenate(
    [arr.reshape(-1,1) for arr in 
     [grad_p_z, div_grad_u_z, buoyancy, u_grad_u_z, vorticity, dvort_dt, lift_tau_u2_z]], 
     axis=1
     ).astype(np.float32)

y1 = duz_dt.reshape(-1,1).astype(np.float32)
print(X1.shape, y1.shape)

X2 = np.concatenate(
    [arr.reshape(-1,1) for arr in 
     [grad_p_x, div_grad_u_x, buoyancy, u_grad_u_x, vorticity, dvort_dt, lift_tau_u2_x]], 
     axis=1
     ).astype(np.float32)

y2 = dux_dt.reshape(-1,1).astype(np.float32)
print(X2.shape, y2.shape)

X3 = np.concatenate(
    [arr.reshape(-1,1) for arr in 
     [div_grad_b, lift_tau_b2, u_grad_b]], 
     axis=1
     ).astype(np.float32)

y3 = db_dt.reshape(-1,1).astype(np.float32)
print(X3.shape, y3.shape)
  
# delete unnecessary variables to free up memory
del buoyancy, div_grad_u, grad_p, velocity, grad_u, du_dt, my_fields
del velocity_x, velocity_z, div_grad_u_x, div_grad_u_z
del grad_p_x, grad_p_z, grad_x_ux, grad_z_ux, grad_x_uz, grad_z_uz, dux_dt, duz_dt
del vorticity, dvort_dt, grad_b, grad_b_x, grad_b_z, u_grad_u_x, u_grad_u_z, u_grad_b
del lift_tau_u2_x, lift_tau_u2_z, lift_tau_u2

# define the model
X1 = sm.add_constant(X1)
model1 = sm.OLS(y1, X1).fit()
# predictions1 = model1.predict(X1)

X2 = sm.add_constant(X2)
model2 = sm.OLS(y2, X2).fit()
# predictions2 = model2.predict(X2)

X3 = sm.add_constant(X3)
model3 = sm.OLS(y3, X3).fit()
# predictions3 = model3.predict(X3)
# %%
# print the statistics
model1.summary()
# %%
model2.summary()
# %%
model3.summary()
# %%
# print vif for the data

vif1 = pd.DataFrame()
vif1["VIF Factor"] = [variance_inflation_factor(X1, i) for i in range(X1.shape[1])]
vif1["features"] = ['grad_p_z', 'div_grad_u_z', 'buoyancy', 'u_grad_u_z',
                    'vorticity', 'dvort_dt']
vif1
# %%
vif2 = pd.DataFrame()
vif2["VIF Factor"] = [variance_inflation_factor(X2, i) for i in range(X2.shape[1])]
vif2["features"] = ['grad_p_x', 'div_grad_u_x', 'buoyancy', 'u_grad_u_x',
                    'vorticity', 'dvort_dt']
vif2
# %%
vif3 = pd.DataFrame()
vif3["VIF Factor"] = [variance_inflation_factor(X3, i) for i in range(X3.shape[1])]
vif3["features"] = ['const', 'div_grad_b', 'lift_tau_b2', 'u_grad_b']
vif3
# %%
