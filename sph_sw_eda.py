#%%
# Perform EDA on the snapshots files data from the spherical shallow water simulations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pysr

import warnings
warnings.filterwarnings('ignore')

from utils import *

# load the fields from the snapshots h5 files
my_fields = [read_snapshots('snapshots/snapshots_s'+str(i)+'.h5') for i in range(1, 37)]

def create_array(x, index, datatype=np.float32):
    '''
    Create an array for the quantity x from the index_th key in my_files
    dictionary and set its dataype to datatype.
    '''
    return np.concatenate([my_fields[i][index][x] for i in range(len(my_fields))],
                                                        axis=0).astype(datatype)

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

for arr, name in [(div_h_u, 'div_h_u'), (div_u, 'div_u'), (grad_h, 'grad_h'),
                  (grad_u, 'grad_u'), (h, 'h'), (lap_h, 'lap_h'),
                  (lap_lap_h, 'lap_lap_h'), (lap_lap_u, 'lap_lap_u'), (u, 'u'),
                  (vort, 'vort'), (zcross_u, 'zcross_u')]:
    print(name, arr.shape)

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
    
# find derivatives wrt time
dux_dt = derivative_wrt_time(u_x, 1)
duz_dt = derivative_wrt_time(u_z, 1)
dh_dt = derivative_wrt_time(h, 1)

# find correlations between the fields
fields = [arr.reshape(-1,1) for arr in [div_h_u, div_u, grad_h_x, grad_h_z, gradx_ux, gradz_ux, gradx_uz,
          gradz_uz, lap_lap_u_x, lap_lap_u_z, u_x, u_z, zcross_u_x, zcross_u_z,
          h, lap_h, lap_lap_h, vort, dux_dt, duz_dt, dh_dt]]

field_names = ['div_h_u', 'div_u', 'grad_h_x', 'grad_h_z', 'gradx_ux', 'gradz_ux',
               'gradx_uz', 'gradz_uz', 'lap_lap_u_x', 'lap_lap_u_z', 'u_x', 'u_z',
               'zcross_u_x', 'zcross_u_z', 'h', 'lap_h', 'lap_lap_h', 'vort',
                'dux_dt', 'duz_dt', 'dh_dt']

# Convert the list of column vectors to a 2D array and transpose it
fields_matrix = np.column_stack(fields).T

# Calculate the correlation matrix
correlations = np.corrcoef(fields_matrix)

plt.figure(figsize=(10, 10))
sns.heatmap(correlations, xticklabels=field_names, yticklabels=field_names, cmap='coolwarm')
plt.show()

x, y = top_k_corrs(correlations, 15)

plot_corr_matrix(correlations, 'Correlation Matrix between the Fields', field_names,
                 to_mark=False, top_k_correlations=x, top_k_indices=y, top_k=15,
                 annot_all=False, annot_top_k=True)

# multiple linear regression to predict dh_dt

X = [arr.flatten() for arr in [lap_lap_h, div_h_u, div_u]]
X = sm.add_constant(np.column_stack(X))
y = dh_dt.flatten()

model = sm.OLS(y, X)
results = model.fit()

print(results.summary()) # R2 = 0.424

# multiple linear regression to predict dux_dt
# dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)

u_grad_u_x = u_x*gradx_ux + u_z*gradz_ux

X = [arr.flatten() for arr in [lap_lap_u_x, grad_h_x, zcross_u_x, u_grad_u_x]]
X = sm.add_constant(np.column_stack(X))
y = dux_dt.flatten()

model = sm.OLS(y, X)
results = model.fit()
print(results.summary()) # R2 = 0.345

# multiple linear regression to predict duz_dt

u_grad_u_z = u_x*gradx_uz + u_z*gradz_uz

X = [arr.flatten() for arr in [lap_lap_u_z, grad_h_z, zcross_u_z, u_grad_u_z]]
X = sm.add_constant(np.column_stack(X))
y = duz_dt.flatten()

model = sm.OLS(y, X)
results = model.fit()
print(results.summary()) # R2 = 0.149