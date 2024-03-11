#%%
# perform eda on the arrays created by Dedalus
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
p_corr_matrix, s_corr_matrix = calculate_correlations(fields)

# Plot the correlation matrices separately
field_names = ['buoyancy', 'pressure', 'velocity_x', 'velocity_z', 'vorticity',
            'div_grad_b', 'div_grad_u_x', 'div_grad_u_z', 'grad_b_x', 'grad_b_z',
            'grad_p_x', 'grad_p_z', 'grad_x_ux', 'grad_x_uz', 'grad_z_ux',
            'grad_z_uz', 'db_dt', 'dp_dt', 'dux_dt', 'duz_dt', 'dvort_dt']

top_5_p, top_5_s = top_k_corrs(p_corr_matrix, s_corr_matrix, k=5)

# print the top 5 most correlated pairs with their correlation values in descending order
for i in range(4, -1, -1):
    print('Top', i+1, 'Pearson correlation:', field_names[top_5_p[0][i]],
          'and', field_names[top_5_p[1][i]], 'with a correlation of', 
            p_corr_matrix[top_5_p[0][i], top_5_p[1][i]])

for i in range(4, -1, -1):
    print('Top', i+1, 'Spearman correlation:', field_names[top_5_s[0][i]],
          'and', field_names[top_5_s[1][i]], 'with a correlation of', 
            s_corr_matrix[top_5_s[0][i], top_5_s[1][i]])

# plot the top 5 most correlated pairs
fig, axes = plt.subplots(5, 1, figsize=(15, 20))
axes = axes.flatten()

for i in range(5):
    axes[i].scatter(fields[top_5_p[0][i]].flatten(),
                    fields[top_5_p[1][i]].flatten(), s=1)
    axes[i].set_xlabel(field_names[top_5_p[0][i]])
    axes[i].set_ylabel(field_names[top_5_p[1][i]])
    axes[i].set_title('Pearson correlation: '
                      +str(p_corr_matrix[top_5_p[0][i],top_5_p[1][i]]))
    
# plt.tight_layout()
plt.show()
#%%
# mark on the correlations heatmap the top 5 most p-correlated pairs
sns.heatmap(p_corr_matrix, cmap='coolwarm', center=0)
plt.title('Pearson Correlation Matrix - Top 5 Most Correlated Pairs')
plt.xticks(ticks=np.arange(0.5, len(fields) + 0.5, 1), labels=field_names, rotation=90)
plt.yticks(ticks=np.arange(0.5, len(fields) + 0.5, 1), labels=field_names, rotation=0)

for i in range(5):
    plt.text(top_5_p[1][i]+0.5, top_5_p[0][i]+0.6, '*', fontsize=12, color='black',
            ha='center', va='center')
plt.show()

# mark on the correlations heatmap the top 5 most s-correlated pairs
sns.heatmap(s_corr_matrix, cmap='coolwarm', center=0)
plt.title('Spearman Correlation Matrix - Top 5 Most Correlated Pairs')
plt.xticks(ticks=np.arange(0.5, len(fields) + 0.5, 1), labels=field_names, rotation=90)
plt.yticks(ticks=np.arange(0.5, len(fields) + 0.5, 1), labels=field_names, rotation=0)

for i in range(5):
    plt.text(top_5_s[1][i]+0.5, top_5_s[0][i]+0.6, '*', fontsize=12, color='black',
            ha='center', va='center')
plt.show()
#%%
# mark on the correlations heatmap the quantities in the first PDE
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

pde_1_pcorr, pde_1_scorr = calculate_correlations([fields[i] for i in pde_1_indices])
pde_2_pcorr, pde_2_scorr = calculate_correlations([fields[i] for i in pde_2_indices])

top_5_pde_1 = top_k_corrs(pde_1_pcorr, pde_1_scorr, k=5)
top_5_pde_2 = top_k_corrs(pde_2_pcorr, pde_2_scorr, k=5)

# mark on the correlations heatmap the top 5 most p-correlated pairs in the first PDE
# FIX THIS PLOT
sns.heatmap(pde_1_pcorr, cmap='coolwarm', center=0)
plt.title('Pearson Correlation Matrix - Top 5 Most Correlated Pairs in PDE 1')
plt.xticks(ticks=np.arange(0.5, len(pde_1_indices) + 0.5, 1), labels=pde_1_q_names, rotation=90)
plt.yticks(ticks=np.arange(0.5, len(pde_1_indices) + 0.5, 1), labels=pde_1_q_names, rotation=0)

for i in range(5):
    plt.text(top_5_pde_1[1][i]+0.5, top_5_pde_1[0][i]+0.6, '*', fontsize=12, color='black',
            ha='center', va='center')
plt.show()
# %%
# mark on the correlations heatmap the top 5 most p-correlated pairs in the first PDE