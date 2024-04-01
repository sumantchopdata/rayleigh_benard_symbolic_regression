#%%
# Perform EDA on the snapshots files data from the spherical shallow water simulations
import numpy as np
from utils import *
import pysr

import warnings
warnings.filterwarnings('ignore')

# load the fields from the snapshots h5 files
my_fields = [read_snapshots('snapshots/snapshots_s'+str(i)+'.h5') for i in range(1, 37)]

def create_array(x, index, datatype=np.float32):
    '''
    Create an array for the quantity x from the index_th key in my_files
    dictionary and set its dataype to datatype.
    '''
    return np.concatenate([my_fields[i][index][x] for i in range(len(my_fields))],
                                                        axis=0).astype(datatype)
# %%
div_h_u = create_array('div_of_h_times_u', 0)
div_u = create_array('div_of_velocity', 1)
grad_h = create_array('grad_of_height', 2)
grad_u = create_array('grad_of_velocity', 3)
h = create_array('height', 4)
lap_h = create_array('lap_of_height', 5)
lap_lap_h = create_array('lap_of_lap_of_height', 6)
lap_lap_u = create_array('lap_of_lap_of_velocity', 7)
u = create_array('velocity', 8)
vort = create_array('vorticity', 9)
zcross_u = create_array('zcross_of_velocity', 10)
# %%
for arr, name in [(div_h_u, 'div_h_u'), (div_u, 'div_u'), (grad_h, 'grad_h'),
                  (grad_u, 'grad_u'), (h, 'h'), (lap_h, 'lap_h'),
                  (lap_lap_h, 'lap_lap_h'), (lap_lap_u, 'lap_lap_u'), (u, 'u'),
                  (vort, 'vort'), (zcross_u, 'zcross_u')]:
    print(name, arr.shape)
# %%
# split the fields in x and z directions
grad_h_x = grad_h[:, 0, :, :]
grad_h_z = grad_h[:, 1, :, :]

gradx_ux = grad_u[:, 0, 0, :, :]
gradz_ux = grad_u[:, 0, 1, :, :]
gradx_uz = grad_u[:, 1, 0, :, :]
gradz_uz = grad_u[:, 1, 1, :, :]

lap_lap_u_x = lap_lap_u[:, 0, :, :]
lap_lap_u_z = lap_lap_u[:, 1, :, :]

u_x = u[:, 0, :, :]
u_z = u[:, 1, :, :]

zcross_u_x = zcross_u[:, 0, :, :]
zcross_u_z = zcross_u[:, 1, :, :]
# %%
for arr, name in [(grad_h_x, 'grad_h_x'), (grad_h_z, 'grad_h_z'),
                  (gradx_ux, 'gradx_ux'), (gradz_ux, 'gradz_ux'),
                  (gradx_uz, 'gradx_uz'), (gradz_uz, 'gradz_uz'),
                  (lap_lap_u_x, 'lap_lap_u_x'), (lap_lap_u_z, 'lap_lap_u_z'),
                  (u_x, 'u_x'), (u_z, 'u_z'), (zcross_u_x, 'zcross_u_x'),
                  (zcross_u_z, 'zcross_u_z'), (div_h_u, 'div_h_u'), (div_u, 'div_u'),
                  (h, 'h'), (lap_h, 'lap_h'), (lap_lap_h, 'lap_lap_h'), (vort, 'vort')]:
    print(name, arr.shape)
    print('min:', arr.min(), 'max:', arr.max())
    print('mean:', arr.mean(), 'std:', arr.std())
    print()
# %%
