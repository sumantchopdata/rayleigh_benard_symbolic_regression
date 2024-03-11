#%%
# perform eda on the arrays created by Dedalus
import numpy as np

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
grad_b = create_array('grad_b', 4)
grad_p = create_array('grad_p', 5)
grad_u = create_array('grad_u', 6)
pressure = create_array('pressure', 9)
velocity = create_array('velocity', 10)
vorticity = create_array('vorticity', 11)

div_grad_u_x = div_grad_u[:, 0, :, :]
div_grad_u_z = div_grad_u[:, 1, :, :]
grad_b_x = grad_b[:, 0, :, :]
grad_b_z = grad_b[:, 1, :, :]
grad_p_x = grad_p[:, 0, :, :]
grad_p_z = grad_p[:, 1, :, :]
grad_x_ux = grad_u[:, 0, 0, :, :]
grad_x_uz = grad_u[:, 0, 1, :, :]
grad_z_ux = grad_u[:, 1, 0, :, :]
grad_z_uz = grad_u[:, 1, 1, :, :]
velocity_x = velocity[:, 0, :, :]
velocity_z = velocity[:, 1, :, :]

db_dt = derivative_wrt_time(buoyancy, 0.25)
dp_dt = derivative_wrt_time(pressure, 0.25)
du_dt = derivative_wrt_time(velocity, 0.25)
dvort_dt = derivative_wrt_time(vorticity, 0.25)

dux_dt = du_dt[:, 0, :, :]
duz_dt = du_dt[:, 1, :, :]

fields = [buoyancy, pressure, velocity_x, velocity_z, vorticity,
          div_grad_b, div_grad_u_x, div_grad_u_z, grad_b_x, grad_b_z,
          grad_p_x, grad_p_z, grad_x_ux, grad_x_uz, grad_z_ux, grad_z_uz,
          db_dt, dp_dt, dux_dt, duz_dt, dvort_dt]

# delete unnecessary arrays
del div_grad_u, grad_b, grad_p, grad_u, velocity, du_dt

# mean_and_sd = pd.DataFrame({'field_name': field_names,
#                             'shape': [field.shape for field in fields],
#                             'mean': [np.mean(field) for field in fields],
#                             'abs_mean': [np.mean(np.abs(field)) for field in fields],
#                             'std': [np.std(field) for field in fields]})

# print(mean_and_sd)

fields = [field.reshape(50,4,256,64).mean(axis=(1)).reshape(50,-1).astype(np.float32)
          for field in fields]

# Calculate correlations
p_corr_matrix = get_pearson_corr(fields)

# Plot the correlation matrices separately
field_names = ['buoyancy', 'pressure', 'velocity_x', 'velocity_z', 'vorticity',
            'div_grad_b', 'div_grad_u_x', 'div_grad_u_z', 'grad_b_x', 'grad_b_z',
            'grad_p_x', 'grad_p_z', 'grad_x_ux', 'grad_x_uz', 'grad_z_ux',
            'grad_z_uz', 'db_dt', 'dp_dt', 'dux_dt', 'duz_dt', 'dvort_dt']
#%%
top_5_p, top_5_indices = top_k_corrs(p_corr_matrix, k=5)
#%%
# print the top 5 most correlated pairs with their correlation values in descending order
for i in range(4, -1, -1):
    print('Top', i+1, 'Pearson correlation:', field_names[top_5_indices[0][i]],
          'and', field_names[top_5_indices[1][i]], 'with a correlation of', 
            p_corr_matrix[top_5_indices[0][i], top_5_indices[1][i]])
#%%
# mark on the correlations heatmap the top 5 most p-correlated pairs
plot_corr_matrix(p_corr_matrix,
                 'Pearson Correlation Matrix - Top 5 Most Correlated Pairs',
                 field_names, to_mark=False, top_k_correlations=top_5_p,
                 top_k_indices=top_5_indices, annot_all=False, annot_top_k=True)
#%%
# mention the quantities in the first and the second PDEs
pde_1_q = ['db_dt', 'div_grad_b', 'velocity_x', 'velocity_z', 'grad_b_x', 'grad_b_z']

pde_1_indices = np.array([field_names.index(field) for field in field_names
                          if field in pde_1_q])
pde_1_q_names = np.array([field_names[i] for i in pde_1_indices])

pde_2_q = ['dux_dt', 'duz_dt', 'div_grad_u_x', 'div_grad_u_z',
           'grad_p_x', 'grad_p_z', 'buoyancy','velocity_x', 'velocity_z',
           'grad_x_ux', 'grad_x_uz', 'grad_z_ux', 'grad_z_uz']

pde_2_indices = np.array([field_names.index(field) for field in field_names
                                                if field in pde_2_q])
pde_2_q_names = [field_names[i] for i in pde_2_indices]

pde_1_corr = get_pearson_corr([fields[i] for i in pde_1_indices])
pde_2_corr = get_pearson_corr([fields[i] for i in pde_2_indices])
#%%
top_5_pde_1_p, top_5_pde_1_indices = top_k_corrs(pde_1_corr, k=5)
top_5_pde_2_p, top_5_pde_2_indices = top_k_corrs(pde_2_corr, k=5)
#%%
# mark on the correlations heatmap the top 5 most p-correlated pairs in the first PDE
plot_corr_matrix(pde_1_corr,
                 'Pearson Correlation Matrix - Top 5 Most Correlated Pairs in PDE 1',
                 pde_1_q_names, to_mark=False, top_k_correlations=top_5_pde_1_p,
                 top_k_indices=top_5_pde_1_indices,
                 annot_all=False, annot_top_k=True)
#%%
# mark on the correlations heatmap the top 5 most p-correlated pairs in the second PDE
plot_corr_matrix(pde_2_corr,
                    'Pearson Correlation Matrix - Top 5 Most Correlated Pairs in PDE 2',
                    pde_2_q_names, to_mark=False, top_k_correlations=top_5_pde_2_p,
                    top_k_indices=top_5_pde_2_indices,
                    annot_all=False, annot_top_k=True)
# %%
# print all the pairs with their correlation values in descending order for the first pde
for i in range(4, -1, -1):
    print('Top', i+1, 'Pearson correlation:', pde_1_q_names[top_5_pde_1_indices[0][i]],
          'and', pde_1_q_names[top_5_pde_1_indices[1][i]], 'with a correlation of',
            pde_1_corr[top_5_pde_1_indices[0][i], top_5_pde_1_indices[1][i]])

# print all the pairs with their correlation values in descending order for the second pde
for i in range(4, -1, -1):
    print('Top', i+1, 'Pearson correlation:', pde_2_q_names[top_5_pde_2_indices[0][i]],
          'and', pde_2_q_names[top_5_pde_2_indices[1][i]], 'with a correlation of',
            pde_2_corr[top_5_pde_2_indices[0][i], top_5_pde_2_indices[1][i]])
#%%
# create the matrices of u_grad_u and u_grad_b and now see what happens
u_grad_u_x = np.multiply(velocity_x, grad_x_ux) + np.multiply(velocity_z, grad_z_ux)
u_grad_u_z = np.multiply(velocity_x, grad_x_uz) + np.multiply(velocity_z, grad_z_uz)
u_grad_b = np.multiply(velocity_x, grad_b_x) + np.multiply(velocity_z, grad_b_z)

new_fields = [u_grad_u_x, u_grad_u_z, u_grad_b]
new_fields = [field.reshape(50,4,256,64).mean(axis=(1)).reshape(50,-1).astype(np.float32)
            for field in new_fields]

fields.extend(new_fields)

field_names = ['buoyancy', 'pressure', 'velocity_x', 'velocity_z', 'vorticity',
            'div_grad_b', 'div_grad_u_x', 'div_grad_u_z', 'grad_b_x', 'grad_b_z',
            'grad_p_x', 'grad_p_z', 'grad_x_ux', 'grad_x_uz', 'grad_z_ux',
            'grad_z_uz', 'db_dt', 'dp_dt', 'dux_dt', 'duz_dt', 'dvort_dt',
            'u_grad_u_x', 'u_grad_u_z', 'u_grad_b']
#%%
# Calculate correlations
p_corr_matrix = get_pearson_corr(fields, field_names) # FIX THIS FUNCTION

# Plot the correlation matrices separately
top_5_p, top_5_indices = top_k_corrs(p_corr_matrix, k=5)

# mark on the correlations heatmap the top 5 most p-correlated pairs
plot_corr_matrix(p_corr_matrix,
                 'Pearson Correlation Matrix - Top 5 Most Correlated Pairs',
                 field_names, to_mark=False, top_k_correlations=top_5_p,
                 top_k_indices=top_5_indices, annot_all=False, annot_top_k=True)
#%%