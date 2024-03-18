#%%
# model 3 - predicting the time derivative of buoyancy
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
div_grad_b = create_array('div_grad_b', 1)
lift_tau_b2 = create_array('lift_tau_b2', 7)

grad_b = create_array('grad_b', 4)
velocity = create_array('velocity', 10)

db_dt = derivative_wrt_time(buoyancy, 0.25)
db_dt = db_dt.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

# perform mean-pooling so that arrays become (25, 64, 32) instead of (200, 256, 64)
buoyancy = buoyancy.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
div_grad_b = div_grad_b.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
lift_tau_b2 = lift_tau_b2.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

grad_b_x = grad_b[:, 0, :, :]
grad_b_z = grad_b[:, 1, :, :]
grad_b_x = grad_b_x.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
grad_b_z = grad_b_z.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

ux = velocity[:, 0, :, :]
uz = velocity[:, 1, :, :]
ux = ux.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)
uz = uz.reshape(50, 4, 128, 2, 32, 2).mean(axis=(1, 3, 5)).reshape(-1, 1)

# Now we need to convert the data to a format that PySR can understand
# We need an X array and a y array.
# We will take the reshaped div_grad_u_x and div_grad_u_z as the y array
# and everything else as the X array.
u_grad_b = ux * grad_b_x + uz * grad_b_z

X = np.concatenate([div_grad_b, lift_tau_b2, u_grad_b], axis=1)
y = db_dt
print(X.shape, y.shape)

# Delete unnecessary variables
del buoyancy, div_grad_b, lift_tau_b2, grad_b, velocity, db_dt, grad_b_x, grad_b_z,
del u_grad_b, ux, uz, my_fields
#%%
# model = pysr.PySRRegressor(binary_operators=["+", "*", "-"],
                        #    use_frequency=False,
                        #    use_frequency_in_tournament=False,
                        #    adaptive_parsimony_scaling=5,
                        #    verbosity=0)
model = pysr.PySRRegressor.from_file('pickled_files/RB_db_dt_without_aps.pkl')
model.fit(X, y)
print("R^2:", model.score(X, y))
print(model.sympy())
# %%