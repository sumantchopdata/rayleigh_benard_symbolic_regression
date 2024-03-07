#%%
# Predicting the derivatives of u
import numpy as np
import pysr

import warnings
warnings.filterwarnings('ignore')

from utils import *

# load the fields from the snapshots h5 files
my_fields = [read_snapshots('snapshots_s'+str(i)+'.h5') for i in range(1, 5)]

def create_array(x, index, datatype=np.float32):
    '''
    Create an array for the quantity x from the index_th key in my_files
    dictionary and set its dataype to datatype.
    '''
    return np.concatenate([my_fields[i][index][x] for i in range(len(my_fields))],
                                                        axis=0).astype(datatype)

buoyancy = create_array('buoyancy', 0)
div_grad_u = create_array('div_grad_u', 2)
lift_tau_u2 = create_array('lift_tau_u2', 7)

grad_p = create_array('grad_p', 4)
velocity = create_array('velocity', 9)

grad_u = create_array('grad_u', 5)

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

grad_ux_x = grad_u[:, 0, 0, :, :]
grad_ux_z = grad_u[:, 0, 1, :, :]
grad_uz_x = grad_u[:, 1, 0, :, :]
grad_uz_z = grad_u[:, 1, 1, :, :]

grad_ux_x = grad_ux_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
grad_ux_z = grad_ux_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
grad_uz_x = grad_uz_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
grad_uz_z = grad_uz_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)

dux_dt = du_dt[:, 0, :, :]
duz_dt = du_dt[:, 1, :, :]

dux_dt = dux_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)
duz_dt = duz_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)

# Now we need to convert the data to a format that PySR can understand
# We need an X array and a y array.
# We will take the reshaped div_grad_u_x and div_grad_u_z as the y array
# and everything else as the X array.
u_grad_u = np.multiply(velocity_x, grad_ux_x) + np.multiply(velocity_z, grad_uz_z)
X = np.concatenate([arr[2:, :, :].reshape(-1,1) for arr in [grad_p_x, grad_p_z, 
                            div_grad_u_x, div_grad_u_z, buoyancy, lift_tau_u2_x, 
                            lift_tau_u2_z, u_grad_u]], axis=1).astype(np.float32)
y = np.concatenate([dux_dt[2:, :, :].reshape(-1,1)], axis=1).astype(np.float32)
print(X.shape, y.shape)
#%%
# Now we can use PySR to find the relationship between X and y
# We are decreasing the values of parsimony and adaptive_parsimony_scaling
# from the default values to make slightly more complicated equations

# delete unnecessary variables to free up memory
del buoyancy, div_grad_u, lift_tau_u2, grad_p, velocity, grad_u, du_dt, my_fields
del velocity_x, velocity_z, div_grad_u_x, div_grad_u_z, lift_tau_u2_x, lift_tau_u2_z
del grad_p_x, grad_p_z, grad_ux_x, grad_ux_z, grad_uz_x, grad_uz_z, dux_dt, duz_dt, u_grad_u
#%%
model = pysr.PySRRegressor(binary_operators=["+", "*", "-", "/", "^"],
                    unary_operators=["exp", "abs", "relu",
                                        "sqrt", "square", "cube",
                                        "log10", "log2", "log",
                                        "sin", "cos", "tan",
                                        "atan", "sinh", "cosh", "tanh",
                                        "sign", "floor", "ceil"],
                #     parsimony=0.0001, use_frequency=False,
                #     use_frequency_in_tournament=True,
                #     adaptive_parsimony_scaling=aps,
                     verbosity=0)

model.fit(X, y)
print("R^2 = ", model.score(X, y))
print(model.sympy())