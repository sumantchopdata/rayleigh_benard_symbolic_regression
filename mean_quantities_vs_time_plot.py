#%%
# plot the mean of the average of the absolute quantities against time to discover the two regimes.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import warnings
warnings.filterwarnings('ignore')

from utils import *

# load the fields from the snapshots h5 files
my_fields = [read_snapshots('RB_snaps/snapshots_1e9_0.2_0.0625/snapshots_s'+str(i)+'.h5') for i in range(1, 7)]

def create_array(x, index, datatype=np.float32):
    '''
    Create an array for the quantity x from the index_th key in my_files
    dictionary and set its dataype to datatype.
    '''
    return np.concatenate([my_fields[i][index][x] for i in range(len(my_fields))],
                                                        axis=0).astype(datatype)

buoyancy = create_array('buoyancy', 0)
# div_grad_b = create_array('div_grad_b', 1)
# div_grad_u = create_array('div_grad_u', 2)
# ez = create_array('ez', 3)
# grad_b = create_array('grad_b', 4)
# grad_p = create_array('grad_p', 5)
# grad_u = create_array('grad_u', 6)
# lift_tau_b2 = create_array('lift_tau_b2', 7)
# lift_tau_u2 = create_array('lift_tau_u2', 8)
pressure = create_array('pressure', 9)
velocity = create_array('velocity', 10)
vorticity = create_array('vorticity', 11)

# db_dt = derivative_wrt_time(buoyancy, 0.25)
# du_dt = derivative_wrt_time(velocity, 0.25)
# dvort_dt = derivative_wrt_time(vorticity, 0.25)
# dp_dt = derivative_wrt_time(pressure, 0.25)

velocity_x = velocity[:, 0, :, :]
velocity_z = velocity[:, 1, :, :]

mean_vorticity = np.mean(np.abs(vorticity), axis=(1, 2))
time = np.arange(0, mean_vorticity.shape[0])
mean_pressure = np.mean(np.abs(pressure), axis=(1, 2))
mean_velocity_x = np.mean(np.abs(velocity_x), axis=(1, 2))
mean_velocity_z = np.mean(np.abs(velocity_z), axis=(1, 2))
mean_buoyancy = np.mean(np.abs(buoyancy), axis=(1, 2))
#%%
# plot these against time
for i, (y, label) in enumerate(zip(
        [mean_vorticity, mean_pressure, mean_velocity_x, mean_velocity_z, mean_buoyancy],
        ['vorticity', 'pressure', 'velocity_x', 'velocity_z', 'buoyancy'])):
    
    fig, ax = plt.subplots()
    ax.plot(time, y)

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter('{x:.0f}')

    ax.xaxis.set_minor_locator(MultipleLocator(5))

    plt.grid(visible=True)
    plt.xlabel('timesteps')
    plt.ylabel('mean of the absolute value')
    plt.title('Mean of the absolute value of {} against time'.format(label))
    plt.show()