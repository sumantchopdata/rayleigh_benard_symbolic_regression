#%%
import numpy as np
import pysr

import warnings
warnings.filterwarnings('ignore')

from utils import *

# load the fields from the snapshots h5 files
my_fields = [read_snapshots('snapshots_s'+str(i)+'.h5') for i in range(1, 5)]

buoyancy = np.concatenate([my_fields[i][0]['buoyancy'] for i in range(4)], axis=0)
div_grad_u = np.concatenate([my_fields[i][2]['div_grad_u'] for i in range(4)], axis=0)

grad_p = np.concatenate([my_fields[i][4]['grad_p'] for i in range(4)], axis=0)
velocity = np.concatenate([my_fields[i][9]['velocity'] for i in range(4)], axis=0)

grad_u = np.concatenate([my_fields[i][5]['grad_u'] for i in range(4)], axis=0)

du_dt = derivative_wrt_time(velocity, 0.25)

# perform mean-pooling so that arrays become (25, 64, 32) instead of (200, 256, 64)
buoyancy = buoyancy.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

div_grad_u_x = div_grad_u[:, 0, :, :]
div_grad_u_z = div_grad_u[:, 1, :, :]

div_grad_u_x = div_grad_u_x.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1,1)
div_grad_u_z = div_grad_u_z.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1,1)

grad_p_x = grad_p[:, 0, :, :]
grad_p_z = grad_p[:, 1, :, :]
grad_p_x = grad_p_x.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
grad_p_z = grad_p_z.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

grad_x_ux = grad_u[:, 0, 0, :, :]
grad_z_ux = grad_u[:, 0, 1, :, :]
grad_x_uz = grad_u[:, 1, 0, :, :]
grad_z_uz = grad_u[:, 1, 1, :, :]

grad_x_ux = grad_x_ux.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
grad_z_ux = grad_z_ux.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
grad_x_uz = grad_x_uz.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
grad_z_uz = grad_z_uz.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

dux_dt = du_dt[:, 0, :, :]
duz_dt = du_dt[:, 1, :, :]

dux_dt = dux_dt.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
duz_dt = duz_dt.reshape(25, 8, 64, 4, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

# Now we need to convert the data to a format that PySR can understand
# We need an X array and a y array.
# We will take the reshaped div_grad_u_x and div_grad_u_z as the y array
# and everything else as the X array.

X = np.concatenate([grad_p_x, grad_p_z, dux_dt, duz_dt, buoyancy,
                    grad_x_ux, grad_z_ux, grad_x_uz, grad_z_uz], axis=1)
y = np.concatenate([div_grad_u_x, div_grad_u_z], axis=1)
print(X.shape, y.shape)
#%%
# Now we can use PySR to find the relationship between X and y
# model = pysr.PySRRegressor(binary_operators=["+", "*", "-", "/"],
#                            unary_operators=["exp", "log", "abs", "sqrt", "square", "cube"],
#                            verbosity=0)
model = pysr.PySRRegressor.from_file('base_model_div_grad_u_2024-02-28_184517.737.pkl')
# model.fit(X, y)
#%%
plotting(model, X, y, 0, 'The truth vs prediction for the x component of Laplacian of u')
plotting(model, X, y, 1, 'The truth vs prediction for the z component of Laplacian of u')