#%%
# plot the mean of the average of the absolute quantities against time to discover the two regimes.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import warnings
warnings.filterwarnings('ignore')

from utils import *

# load the fields from the snapshots h5 files
mean_vorts, mean_pressures, mean_vel_xs, mean_vel_zs, mean_buoys = [], [], [], [], []
for folder in os.listdir('RB_snaps/'):
    rayleigh = float(folder.split('_')[1])
    prandtl = float(folder.split('_')[2])

    my_fields = [read_snapshots('RB_snaps/'+folder+'/snapshots_s'+str(i)+'.h5')
                 for i in range(1, len(os.listdir('RB_snaps/'+folder+'/'))+1)]

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

    mean_vort = np.mean(np.abs(vorticity), axis=(1, 2))
    time = np.arange(0, mean_vort.shape[0])
    mean_pressure = np.mean(np.abs(pressure), axis=(1, 2))
    mean_vel_x = np.mean(np.abs(velocity_x), axis=(1, 2))
    mean_vel_z = np.mean(np.abs(velocity_z), axis=(1, 2))
    mean_buoy = np.mean(np.abs(buoyancy), axis=(1, 2))

    mean_vorts.append((mean_vort, rayleigh, prandtl))
    mean_pressures.append((mean_pressure, rayleigh, prandtl))
    mean_vel_xs.append((mean_vel_x, rayleigh, prandtl))
    mean_vel_zs.append((mean_vel_z, rayleigh, prandtl))
    mean_buoys.append((mean_buoy, rayleigh, prandtl))

# plot these against time such that all the plots for the same mean quantity are on the same plot
# and the different plots are in different figures.

plotting_tuple = [(mean_vel_xs, 'velocity_x'), (mean_vel_zs, 'velocity_z')]

for mean_quantity, label in plotting_tuple:
    fig, ax = plt.subplots()
    for y, r, p in mean_quantity:
        if (r, p) != (50000000, 0.2) and (r, p) != (50000000, 0.03): # these are the two outliers
            y = np.nan_to_num(y, nan=0)
            ax.plot(y[1:200], label=f'Rayleigh: {r:.0e}, Prandtl: {p}')

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter('{x:.0f}')

    ax.xaxis.set_minor_locator(MultipleLocator(5))

    plt.grid(visible=True)
    plt.xlabel('time')
    # plt.yscale('log')
    plt.ylabel('Mean of the absolute value')
    plt.title('Mean of the absolute value of {} against time'.format(label))
    plt.legend()
    plt.savefig('mean_'+label+'_vs_time_plot_for_different_r_and_p.png')
    plt.show()