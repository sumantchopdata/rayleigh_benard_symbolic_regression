#%%
# RB_model_5 for du_dt
import numpy as np
import pysr

import warnings
warnings.filterwarnings('ignore')

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
# div_grad_b = create_array('div_grad_b', 1)
div_grad_u = create_array('div_grad_u', 2)
ez = create_array('ez', 3)
# grad_b = create_array('grad_b', 4)
grad_p = create_array('grad_p', 5)
grad_u = create_array('grad_u', 6)
# lift_tau_b2 = create_array('lift_tau_b2', 7)
lift_tau_u2 = create_array('lift_tau_u2', 8)
pressure = create_array('pressure', 9)
velocity = create_array('velocity', 10)
vorticity = create_array('vorticity', 11)

du_dt = derivative_wrt_time(velocity, 0.25)
dvort_dt = derivative_wrt_time(vorticity, 0.25)
dp_dt = derivative_wrt_time(pressure, 0.25)

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

u_grad_u_x = np.multiply(velocity_x, grad_x_ux) + np.multiply(velocity_z, grad_z_ux)
u_grad_u_z = np.multiply(velocity_x, grad_x_uz) + np.multiply(velocity_z, grad_z_uz)

dp_dt = dp_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)
dvort_dt = dvort_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)

vorticity = vorticity.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)

# standardise all the arrays to have zero mean and unit variance
def standardise(arr):
    return (arr - np.mean(arr)) / np.std(arr)

for arr in [dux_dt, duz_dt, div_grad_u_x, div_grad_u_z, dvort_dt, vorticity, 
            buoyancy, u_grad_u_x, u_grad_u_z]:
    arr = standardise(arr)

X = np.concatenate(
    [arr[4:, ...].reshape(-1,1) for arr in 
        [dux_dt, duz_dt, div_grad_u_x, div_grad_u_z, dvort_dt, vorticity, 
        buoyancy, u_grad_u_x, u_grad_u_z]],
     axis=1
     ).astype(np.float32)

y = np.concatenate([arr[4:, :, :].reshape(-1,1) for arr in
                        [grad_p_x, grad_p_z]],
                    axis=1).astype(np.float32)
print(X.shape, y.shape)

# delete unnecessary variables to free up memory
del buoyancy, div_grad_u, lift_tau_u2, grad_p, velocity, grad_u, du_dt, my_fields
del velocity_x, velocity_z, div_grad_u_x, div_grad_u_z, lift_tau_u2_x, lift_tau_u2_z
del grad_p_x, grad_p_z, grad_x_ux, grad_z_ux, grad_x_uz, grad_z_uz, dux_dt, duz_dt
del u_grad_u_x, u_grad_u_z, vorticity, pressure, dp_dt, dvort_dt, ez

# model = pysr.PySRRegressor(binary_operators=["+", "*", "-"], verbosity=0,
#                            use_frequency=False,
#                            use_frequency_in_tournament=False,
#                            adaptive_parsimony_scaling=1)
model = pysr.PySRRegressor.from_file('RB_model_5_grad_p.pkl')
model.fit(X, y)
print("R^2 = ", model.score(X, y)) # 0.3895 including grad_p_x and grad_p_z
print(model.sympy())