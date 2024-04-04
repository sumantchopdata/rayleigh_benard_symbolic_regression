#%%
# predict the derivative of vorticity using PySR
import numpy as np
import pysr
from sklearn.metrics import r2_score

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

# buoyancy = create_array('buoyancy', 0)
# div_grad_b = create_array('div_grad_b', 1)
# div_grad_u = create_array('div_grad_u', 2)
# ez = create_array('ez', 3)
# grad_b = create_array('grad_b', 4)
# grad_p = create_array('grad_p', 5)
grad_u = create_array('grad_u', 6)
# lift_tau_b2 = create_array('lift_tau_b2', 7)
# lift_tau_u2 = create_array('lift_tau_u2', 8)
# pressure = create_array('pressure', 9)
velocity = create_array('velocity', 10)
vorticity = create_array('vorticity', 11)

# db_dt = derivative_wrt_time(buoyancy, 0.25)
du_dt = derivative_wrt_time(velocity, 0.25)
dvort_dt = derivative_wrt_time(vorticity, 0.25)
# dp_dt = derivative_wrt_time(pressure, 0.25)

# average the fields in the x and z directions and in time
grad_x_ux = grad_u[:, 0, 0, :, :]
grad_z_ux = grad_u[:, 0, 1, :, :]
grad_x_uz = grad_u[:, 1, 0, :, :]
grad_z_uz = grad_u[:, 1, 1, :, :]

grad_x_ux = grad_x_ux.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
grad_z_ux = grad_z_ux.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
grad_x_uz = grad_x_uz.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)
grad_z_uz = grad_z_uz.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float16)

vorticity = vorticity.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)

dt_grad_x_uz = derivative_wrt_time(grad_x_uz, 0.25)
dt_grad_z_ux = derivative_wrt_time(grad_z_ux, 0.25)

# This is the actual formula for the derivative of vorticity
dvort_dt_actual = dt_grad_z_ux - dt_grad_x_uz

dvort_dt = dvort_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1,3,5)).astype(np.float32)

X = np.concatenate(
    [arr.reshape(-1,1) for arr in 
     
     [dt_grad_x_uz, dt_grad_z_ux, vorticity]], 
     
     axis=1).astype(np.float32)

y = dvort_dt.reshape(-1,1).astype(np.float32)
print(X.shape, y.shape)
#%%
# create the symbolic regressor
# model = pysr.PySRRegressor(binary_operators=["+", "*", "-"], verbosity=0)

# Or you can call the pickled file by uncommenting the line below and commenting the line above
model = pysr.PySRRegressor.from_file('pickled_files/RB_dvort_dt_2024-03-20_150537.197.pkl')

model.fit(X, y)
print('R^2: ', model.score(X, y))
print(model.sympy) # -0.5x0 + 0.5x1