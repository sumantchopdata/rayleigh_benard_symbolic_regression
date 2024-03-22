#%%
# Check for one timestep if the equations work for both buoyancy and the velocity

import numpy as np
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
div_grad_u = create_array('div_grad_u', 2)
ez = create_array('ez', 3)
grad_b = create_array('grad_b', 4)
grad_p = create_array('grad_p', 5)
grad_u = create_array('grad_u', 6)
lift_tau_b2 = create_array('lift_tau_b2', 7)
lift_tau_u2 = create_array('lift_tau_u2', 8)
# pressure = create_array('pressure', 9)
velocity = create_array('velocity', 10)
# vorticity = create_array('vorticity', 11)

du_dt = derivative_wrt_time(velocity, 0.25)
# dvort_dt = derivative_wrt_time(vorticity, 0.25)
# dp_dt = derivative_wrt_time(pressure, 0.25)
db_dt = derivative_wrt_time(buoyancy, 0.25)

# perform mean-pooling so that arrays become (50, 128, 32) instead of (200, 256, 64)
buoyancy = buoyancy.astype(np.float32)

div_grad_b = div_grad_b.astype(np.float32)

velocity_x = velocity[:, 0, :, :].astype(np.float32)
velocity_z = velocity[:, 1, :, :].astype(np.float32)

div_grad_u_x = div_grad_u[:, 0, :, :].astype(np.float32)
div_grad_u_z = div_grad_u[:, 1, :, :].astype(np.float32)

lift_tau_u2_x = lift_tau_u2[:, 0, :, :].astype(np.float32)
lift_tau_u2_z = lift_tau_u2[:, 1, :, :].astype(np.float32)

div_grad_u_x = div_grad_u_x.astype(np.float16)
div_grad_u_z = div_grad_u_z.astype(np.float16)

grad_p_x = grad_p[:, 0, :, :].astype(np.float32)
grad_p_z = grad_p[:, 1, :, :].astype(np.float32)

grad_x_ux = grad_u[:, 0, 0, :, :].astype(np.float32)
grad_z_ux = grad_u[:, 0, 1, :, :].astype(np.float32)
grad_x_uz = grad_u[:, 1, 0, :, :].astype(np.float32)
grad_z_uz = grad_u[:, 1, 1, :, :].astype(np.float32)

dux_dt = du_dt[:, 0, :, :].astype(np.float32)
duz_dt = du_dt[:, 1, :, :].astype(np.float32)

ez = ez.reshape((200,2))
ez = np.tile(ez[:, np.newaxis, :], (1,256,32)).astype(np.float32)

grad_b_x = grad_b[:, 0, :, :].astype(np.float32)
grad_b_z = grad_b[:, 1, :, :].astype(np.float32)

u_grad_u_x = (np.multiply(velocity_x, grad_x_ux) + np.multiply(velocity_z, grad_z_ux)).astype(np.float32)
u_grad_u_z = (np.multiply(velocity_x, grad_x_uz) + np.multiply(velocity_z, grad_z_uz)).astype(np.float32)

buoyancy_times_ez = np.multiply(buoyancy, ez).astype(np.float32)
u_grad_b = (velocity_x * grad_b_x + velocity_z * grad_b_z).astype(np.float32)

# standardise the arrays to be between 0 and 1
def standardise(arr):
    arr -= arr.min()
    arr /= arr.max()
    return arr

for arr in [div_grad_b, lift_tau_b2, u_grad_b,
            div_grad_u_x, div_grad_u_z, grad_p_x, grad_p_z, lift_tau_u2_x,
            lift_tau_u2_z, u_grad_u_x, u_grad_u_z, buoyancy_times_ez]:
    arr = standardise(arr)

RHS1 = 0.0007071*div_grad_b - lift_tau_b2 - u_grad_b

LHS1 = buoyancy/0.25

norm1 = [np.linalg.norm(RHS1[i] - LHS1[i])/np.linalg.norm(LHS1[i]) for i in range(1,50,1)]
plt.hist(norm1)
plt.show()

RHS2 = 0.0007071*div_grad_u_x - grad_p_x + buoyancy_times_ez - lift_tau_u2_x - u_grad_u_x
LHS2 = velocity_x/0.25

norm2 = [np.linalg.norm(RHS2[i] - LHS2[i])/np.linalg.norm(LHS2[i]) for i in range(1,50,1)]
plt.hist(norm2)
plt.show()

RHS3 = 0.0007071*div_grad_u_z - grad_p_z - buoyancy_times_ez - lift_tau_u2_z - u_grad_u_z
LHS3 = velocity_z/0.25

norm3 = [np.linalg.norm(RHS3[i] - LHS3[i])/np.linalg.norm(LHS3[i]) for i in range(1,50,1)]
plt.hist(norm3)
plt.show()

for arr in [norm1, norm2, norm3]:
    print(np.mean(arr), np.std(arr))
