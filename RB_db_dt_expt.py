#%%
# Experiment with random noise added to the data on the db/dt model

import numpy as np
import pysr
import matplotlib.pyplot as plt

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
# Add random noise to the model and see how the R2 score changes

R2_noisy_list, model_sympys_list = [], []

noise_list = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.0125,
              0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06,
              0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275,
              0.3, 0.35, 0.4, 0.45, 0.5]

for noise in noise_list:
    X_noisy = X + np.random.normal(0, noise, X.shape)
    y_noisy = y + np.random.normal(0, noise, y.shape)

    model = pysr.PySRRegressor(binary_operators=["+", "*", "-"], maxsize=7,
                               bumper=True, elementwise_loss='L1DistLoss()',
                               populations = 36, batching=True, verbosity=0)
    model.fit(X_noisy, y_noisy)
    
    print('For noise =', noise, 'the best model R2 score is ',
          model.score(X_noisy, y_noisy))
    R2_noisy_list.append(model.score(X_noisy, y_noisy))

    print('The best model is', model.sympy(), '\n')
    model_sympys_list.append(model.sympy())

    del model, X_noisy, y_noisy

# print(R2_noisy_list)
# print(model_sympys_list)
#%%    
# plot the R2 scores for different values of noise

plt.scatter(noise_list, R2_noisy_list)
# for i, txt in enumerate(model_sympys_list):
#     plt.annotate(txt, (noise_list[i], R2_noisy_list[i]))

plt.plot(noise_list, R2_noisy_list)

plt.xlabel('Noise')
plt.xscale('log')
plt.ylabel('R2 score')
# plt.yscale('log')

plt.title('R2 score vs Noise for db/dt model with random noise added')
plt.savefig('R2_vs_noise_for_db_dt.png')
plt.show()
# %%