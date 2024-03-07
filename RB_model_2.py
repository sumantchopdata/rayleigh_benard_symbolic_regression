#%%
# model 2 - predicting buoyancy
import numpy as np
import pysr

import warnings
warnings.filterwarnings('ignore')

from utils import *

# load the fields from the snapshots h5 files
my_fields = [read_snapshots('snapshots_s'+str(i)+'.h5') for i in range(1, 5)]

buoyancy = np.concatenate([my_fields[i][0]['buoyancy'] for i in range(4)], axis=0)
div_grad_u = np.concatenate([my_fields[i][2]['div_grad_u'] for i in range(4)], axis=0)
lift_tau_u2 = np.concatenate([my_fields[i][7]['lift_tau_u2'] for i in range(4)], axis=0)

grad_p = np.concatenate([my_fields[i][4]['grad_p'] for i in range(4)], axis=0)
velocity = np.concatenate([my_fields[i][9]['velocity'] for i in range(4)], axis=0)

grad_u = np.concatenate([my_fields[i][5]['grad_u'] for i in range(4)], axis=0)

du_dt = derivative_wrt_time(velocity, 0.25)

# perform mean-pooling so that arrays become (25, 64, 32) instead of (200, 256, 64)
buoyancy = buoyancy.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

div_grad_u_x = div_grad_u[:, 0, :, :]
div_grad_u_z = div_grad_u[:, 1, :, :]

lift_tau_u2_x = lift_tau_u2[:, 0, :, :]
lift_tau_u2_z = lift_tau_u2[:, 1, :, :]

lift_tau_u2_x = lift_tau_u2_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
lift_tau_u2_z = lift_tau_u2_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

div_grad_u_x = div_grad_u_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1,1)
div_grad_u_z = div_grad_u_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1,1)

grad_p_x = grad_p[:, 0, :, :]
grad_p_z = grad_p[:, 1, :, :]
grad_p_x = grad_p_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
grad_p_z = grad_p_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

grad_ux_x = grad_u[:, 0, 0, :, :]
grad_ux_z = grad_u[:, 0, 1, :, :]
grad_uz_x = grad_u[:, 1, 0, :, :]
grad_uz_z = grad_u[:, 1, 1, :, :]

grad_ux_x = grad_ux_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
grad_ux_z = grad_ux_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
grad_uz_x = grad_uz_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
grad_uz_z = grad_uz_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

dux_dt = du_dt[:, 0, :, :]
duz_dt = du_dt[:, 1, :, :]

dux_dt = dux_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
duz_dt = duz_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

# Now we need to convert the data to a format that PySR can understand
# We need an X array and a y array.
# We will take the reshaped div_grad_u_x and div_grad_u_z as the y array
# and everything else as the X array.

X = np.concatenate([grad_p_x, grad_p_z, div_grad_u_x, div_grad_u_z,
                    grad_ux_x, grad_ux_z, grad_uz_x, grad_uz_z,
                    lift_tau_u2_x, lift_tau_u2_z, dux_dt, duz_dt], axis=1)
y = np.concatenate([buoyancy], axis=1)
print(X.shape, y.shape)

# Now we can use PySR to find the relationship between X and y
# model = pysr.PySRRegressor(binary_operators=["+", "*", "-", "/"],
#                            unary_operators=["exp", "log", "abs", "sqrt", "square", "cube"],
#                            verbosity=0)
model = pysr.PySRRegressor.from_file('RB_model_2_buoyancy_2024-02-29_150813.613.pkl')
# model.fit(X, y)
#%%
plotting(model, X, y, 0, 'The truth vs prediction for buoyancy', remove_outliers=False)