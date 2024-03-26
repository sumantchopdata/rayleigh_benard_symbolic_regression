#%%
# Experiment with parsimony parameter and the adaptive_parsimony_scaling parameter
# on the db/dt model

import numpy as np
import pysr
import matplotlib.pyplot as plt

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
del u_grad_b, ux, uz, my_fields, create_array
#%%
# Perform symbolic regression in two different for loops
# Since the parameters both do not work together

# First loop
# The default value of parsimony is 0.0032, but very small values give similar results as well

pars_list = [0.000001, 0.000005, 0.00001]

R2_pars_list = []

def run_pars_loop(pars_list, R2_pars_list, X, y):
    for pars in pars_list:
        model = pysr.PySRRegressor(
            binary_operators=["+", "-", "*"],
            parsimony = pars,
            verbosity = 0)
        model.fit(X, y)
        print('For parsimony =', pars, 'the best model R2 score is ', model.score(X, y))
        R2_pars_list.append(model.score(X, y))

        # Here, in model.sympy(), x0 is div_grad_b, x1 is lift_tau_b2 and x2 is u_grad_b
        print('The best model is', model.sympy(), '\n')
        del model

run_pars_loop(pars_list, R2_pars_list, X, y)
#%%
# Second loop
# The default value of adaptive_parsimony_scaling is 20
    
aps_list = [1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 200, 500, 1000]

R2_aps_list = []

def run_aps_loop(aps_list, R2_aps_list, X, y):
    for aps in aps_list:
        model = pysr.PySRRegressor(
            binary_operators=["+", "-", "*"],
            adaptive_parsimony_scaling = aps,
            verbosity = 0,
            use_frequency=False,
            use_frequency_in_tournament=False)
        
        model.fit(X, y)
        print('For adaptive_parsimony_scaling =', aps,
              'the best model R2 score is ', model.score(X, y))
        R2_aps_list.append(model.score(X, y))

        # Here, in model.sympy(), x0 is div_grad_b, x1 is lift_tau_b2 and x2 is u_grad_b
        print('The best model is', model.sympy(), '\n')
        del model

run_aps_loop(aps_list, R2_aps_list, X, y)

# plot the R2 scores for different values of adaptive_parsimony_scaling
plt.plot(aps_list, R2_aps_list)
plt.xlabel('Adaptive Parsimony Scaling')
plt.ylabel('R2 score')
plt.title('R2 score vs Adaptive Parsimony Scaling')
plt.show()
#%%
# Add random noise to the model and see how the R2 score changes

# Add different levels of noise to the X array to see the final output
R2_noisy_list = []
noise_list = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
for noise in noise_list:
    X_noisy = X + np.random.normal(0, 0.1, X.shape)
    y_noisy = y + np.random.normal(0, 0.1, y.shape)

    model = pysr.PySRRegressor(binary_operators=["+", "-", "*"], verbosity = 0)
    model.fit(X_noisy, y_noisy)
    print('For noise =', noise, 'the best model R2 score is ', model.score(X_noisy, y_noisy))
    print('The best model is', model.sympy(), '\n')
    del model
    
# plot the R2 scores for different values of parsimony
plt.plot(noise_list, R2_noisy_list)
plt.xlabel('Noise')
plt.ylabel('R2 score')
plt.title('R2 score vs Noise')
plt.show()
# %%