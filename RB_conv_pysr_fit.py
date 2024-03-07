#%%
# import the h5 files and try to fit the data
import os
import pysr
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import tracemalloc # to track the memory usage

from utils import *

today = '2024-02-01'

# load the fields from the snapshots h5 files
my_fields = [read_snapshots('snapshots_s'+str(i)+'.h5') for i in range(1, 5)]

gradient_file_list = os.listdir(os.path.join('my_results_h5_files', today))
gradient_file_list = [f for f in gradient_file_list if 'gradients' in f]
grad_field_names = [f.split('_')[1] for f in gradient_file_list]

to_keep = ['u', 'p']
grad_field_names = [items for items in grad_field_names if items in to_keep]
gradient_file_list = [items for items in gradient_file_list if items.split('_')[1] in to_keep]

gradients = []
for i, f in enumerate(gradient_file_list):
    k = grad_field_names[i]
    v = load_gradients(os.path.join('my_results_h5_files', today, f))
    gradients.append({k:v})
#%%
# Now gradients is a list of dictionaries, each dictionary contains the gradients of a field
# like tauu1, tauu2, u and p.
# The key is the name of the field and the value is a dictionary containing 
# gradient_x and gradient_z as keys and the gradients as values.

# wait we need to convert the data to a format that PySR can understand
# We need an X array and a y array.
# We will take the bouyancy as the y array and everything else as the X array.

db_dt = derivative_wrt_time(my_fields[0]['buoyancy'], my_fields[0]['sim_time'])
du_dt = derivative_wrt_time(my_fields[0]['vorticity'], my_fields[0]['sim_time'])
#%%
X = np.concatenate([*[my_fields[i]['vorticity'] for i in range(4)], db_dt, du_dt,
                    g], axis=1)

y = my_fields[0]['buoyancy']
#%%
model = pysr.PySRRegressor(binary_operators=["+", "*", "-", "/"],
                           unary_operators=["square", "cube", "sqrt"],
                           verbosity=0)

# del gradient_file_list, grad_field_names, gradients, my_fields, db_dt, du_dt
#%%
tracemalloc.start()

print('-'*50)

print('For the all z slices')
coord_z = 100
model.fit(X[:, coord_z, :], y[:, coord_z, :])

# print the mse between y and predictions
print('The relative mse is: ', mean_squared_error(y[:, coord_z, :],
                        model.predict(X[:, coord_z, :]))/np.mean(y[:, coord_z, :]**2))

with open(f'pysr_equations_z_time_unpooled_{coord_z}.txt', 'w') as f:
    for eqn in [str(expr) for expr in model.sympy()]:
        f.write(eqn+'\n')
#%%
plot_comparative_time_heatmaps(X[:, coord_z, :], y[:, coord_z, :], coord_z, model,
                               is_x=False, is_pooled=False)

# print('memory usage was as follows:')
# print(tracemalloc.get_traced_memory())
# tracemalloc.stop()
#%%
print('-'*50)

tracemalloc.start()
print('For the all the x slices')
coord_x = 50
model.fit(X[:, :, coord_x], y[:, :, coord_x])

# print the mse between y and predictions
print('The relative mse is: ', mean_squared_error(y[:, :, coord_x],
                model.predict(X[:, :, coord_x]))/np.mean(y[:, :, coord_x]**2))

with open(f'pysr_equations_x_time_unpooled_{0}.txt'.format(coord_x), 'w') as f:
    for eqn in [str(expr) for expr in model.sympy()]:
        f.write(eqn+'\n')

plot_comparative_time_heatmaps(X[:, :, coord_x], y[:, :, coord_x], coord_x, model,
                               is_x=True, is_pooled=False)

print(tracemalloc.get_traced_memory())
tracemalloc.stop()
print('-'*50)
#%%
# Perform mean pooling on the X and y arrays
# Reshape the array to introduce an extra dimension and then take the mean along these new axes
X = X.reshape(int(X.shape[0]), int(X.shape[1]/2), 2, int(X.shape[2]/2), 2).mean(axis=(2,4))
y = y.reshape(int(y.shape[0]), int(y.shape[1]/2), 2, int(y.shape[2]/2), 2).mean(axis=(2,4))

X = X.reshape(int(X.shape[0]), int(X.shape[1]/2), 2, int(X.shape[2])).mean(axis=(2))
# Now the shapes are (50, 448, 32) and (50, 128, 32).
#%%
print('For the all the z slices of the pooled data')
model.fit(X[:, coord_z, :], y[:, coord_z, :])

# print the mse between y and predictions
print('The relative mse is: ', mean_squared_error(y[:, coord_z, :],
                        model.predict(X[:, coord_z, :]))/np.mean(y[:, coord_z, :]**2))

with open(f'pysr_equations_z_time_pooled_{coord_z}.txt', 'w') as f:
    for eqn in [str(expr) for expr in model.sympy()]:
        f.write(eqn+'\n')

plot_comparative_time_heatmaps(X[:, coord_z, :], y[:, coord_z, :], coord_z, model,
                               is_x=False, is_pooled=True)

print('-'*50)
#%%
tracemalloc.start()
coord_x = 30
print('For the all the x slices of the pooled data')
model.fit(X[:, :, coord_x], y[:, :, coord_x])

# print the mse between y and predictions
print('The relative mse is: ', mean_squared_error(y[:, :, coord_x],
                        model.predict(X[:, :, coord_x]))/np.mean(y[:, :, coord_x]**2))

with open(f'pysr_equations_x_time_pooled_{coord_x}.txt', 'w') as f:
    for eqn in [str(expr) for expr in model.sympy()]:
        f.write(eqn+'\n')

plot_comparative_time_heatmaps(X[:, :, coord_x], y[:, :, coord_x], coord_x, model,
                               is_x=True, is_pooled=True)

print('memory usage was as follows:')

print(tracemalloc.get_traced_memory())
tracemalloc.stop()
#%%
plot_comparative_space_heatmaps(X[:, :, coord_x], y[:, :, coord_x], coord_x, model, is_pooled=True)
# %%
