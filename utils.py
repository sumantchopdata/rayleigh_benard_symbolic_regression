import h5py
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

# define the function to read the snapshot h5 files and return the data
def read_snapshots(file_name):    
    f = h5py.File(os.path.join('snapshots', file_name), 'r')
    
    tasks = f['tasks']
    fields = []

    for k in tasks.keys():
        fields.append({k: tasks[k][...]})

    f.close()
    return fields

# Define a function to return the derivative of an array with respect to time.
def derivative_wrt_time(array, timesteps):
    # if time is a float, directly use it as the time step
    if type(timesteps) == float:
        dt = timesteps

    elif type(timesteps) == np.ndarray or type(timesteps) == list:
        dt = np.mean([timesteps[i+1] - timesteps[i] for i in range(len(timesteps)-1)])

    else:
        dt = 0.25
    return np.array(np.gradient(array, dt, axis=0))

def plot_comparative_time_heatmaps(X, y, coordinate, model, is_x = True, is_pooled=False):
    '''
    make side by side heatmaps of both the actual and the predicted array slices
    against the timesteps
    '''
    z_label, is_row = '', ''

    if is_x:
        z_label = 'pooled x-axis_values' if is_pooled else 'x-axis_values'
        is_row = 'row'

    else:
        z_label = 'pooled z-axis_values' if is_pooled else 'z_axis_values'
        is_row = 'column'

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    plt.imshow(y.T, aspect='auto', cmap='viridis')
    plt.xlabel('timesteps')
    plt.ylabel(z_label)
    plt.colorbar()
    
    plt.title('Actual heatmap')
    
    plt.subplot(1, 2, 2)
    plt.imshow(model.predict(X).T, aspect='auto', cmap='viridis')
    plt.xlabel('timesteps')
    plt.ylabel(z_label)
    plt.colorbar()
    
    plt.title('Predicted heatmap')

    rel_mse = mean_squared_error(model.predict(X),y)/np.mean(y**2)
    heatmap = 'pooled heatmaps' if is_pooled else 'heatmaps'

    plt.suptitle(f'Plot of the actual vs the predicted {heatmap} of the {coordinate}th {is_row} using PySR\n \
                 Relative MSE = {"{:.3e}".format(rel_mse)}')
    
    plt.savefig(f'{heatmap}_{coordinate}_{is_row}', dpi=300)
    plt.show()

def plot_comparative_space_heatmaps(X, y, timestep, is_pooled=False):
    '''
    make the side by side heatmaps of both the actual and the predicted arrays
    at a particular timestep against the spatial coordinates
    '''
    
    plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    plt.subplot(1, 2, 1)

    plt.imshow(y, aspect='auto', cmap='viridis')
    plt.xlabel('x-axis_values')
    plt.ylabel('z-axis_values')
    plt.colorbar()
    plt.title('Actual heatmap')

    plt.subplot(1, 2, 2)
    plt.imshow(X, aspect='auto', cmap='viridis')
    plt.xlabel('x-axis_values')
    plt.ylabel('z-axis_values')

    plt.title('Predicted heatmap')
    plt.colorbar()

    R_2 = r2_score(X, y)
    heatmap = 'pooled heatmaps' if is_pooled else 'heatmaps'

    plt.suptitle(f'Plot of the actual vs the predicted {heatmap} \
                    at the {timestep}th timestep using PySR\n \
                    R^2 score = {"{:.3e}".format(R_2)}')
    
    plt.savefig(f'{heatmap}_{timestep}_{"pooled" if is_pooled else "unpooled"}.png', dpi=300)
    plt.show()

def plotting(model, X, y, i, title):
    '''
    Plot the truth vs prediction plots for the ith column of y
    and model.predict(X). The title of the plot is title.
    '''
    num_vars_to_plot = y.shape[1]

    if num_vars_to_plot == 1:
        model.predict(X)
        X_new, y_new = model.predict(X).reshape(-1, 1), y.reshape(-1, 1)
    
    else:
        X_new, y_new = model.predict(X)[:,i], y[:,i]

    y_min, y_max = np.min(y_new), np.max(y_new)
    x_min, x_max = np.min(X_new), np.max(X_new)
    minimum, maximum = min(x_min, y_min), max(x_max, y_max)

    plt.scatter(y_new, X_new)
    plt.xlabel('Truth')
    plt.ylabel('Prediction')

    # plot the y = x line
    plt.plot(np.linspace(minimum, maximum, 100), np.linspace(minimum, maximum, 100), 'r')
    plt.title(title + '\n R^2 value is ' + str(r2_score(y_new, X_new)))

    plt.savefig(title+'.png')
    plt.show()
    plt.close()

    print('The relative mse is: ', mean_squared_error(y_new, X_new)/np.mean(y_new**2))

def manage_files():
    '''
    Put all the model saved files in a folder called pickled_files
    '''
    endings = ('.pkl', '.csv', '.out1', '.out2', '.bkup')
    if not os.path.exists('pickled_files'):
        os.mkdir('pickled_files')
        
    for file in os.listdir():
        if os.path.isfile(file) and (file.startswith('hall_of_fame') or file.endswith(endings)):
            try:
                shutil.move(file, 'pickled_files')
            except Exception as e:
                print(f"Error occurred while moving file {file}: {str(e)}")
            
# Function to calculate Pearson and Spearman correlations of all the arrays in array_list
def calculate_correlations(array_list):
    n = len(array_list)
    p_corr_matrix = np.zeros((n, n))
    s_corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            array1, array2 = array_list[i].flatten(), array_list[j].flatten()
            pearson_corr, _ = pearsonr(array1, array2)
            spearman_corr, _ = spearmanr(array1, array2)
            
            p_corr_matrix[i, j] = pearson_corr
            s_corr_matrix[i, j] = spearman_corr
            
            if i != j:
                p_corr_matrix[j, i] = pearson_corr
                s_corr_matrix[j, i] = spearman_corr

    return p_corr_matrix, s_corr_matrix

def top_k_corrs(pcm, scm, k=10):
    '''
    Return the top k most correlated pairs (either positive or negative)
    from the Pearson and Spearman correlation matrices pcm and scm.
    We exclude self-correlations and duplicate pairs. The indices of the
    top k most correlated pairs are returned in ascending order.
    '''
    abs_pcm = np.abs(pcm)
    abs_scm = np.abs(scm)

    # set the diagonal and lower triangle to zero
    # to exclude self-correlations and duplicate pairs
    np.fill_diagonal(abs_pcm, 0)
    np.fill_diagonal(abs_scm, 0)

    triu_indices = np.triu_indices(abs_pcm.shape[0], 1)
    abs_pcm[triu_indices] = 0
    abs_scm[triu_indices] = 0

    # find the top 10 most correlated pairs
    top_k_p = np.argpartition(abs_pcm, -k, axis=None)[-k:]
    top_k_s = np.argpartition(abs_scm, -k, axis=None)[-k:]

    # convert the indices to 2D
    top_k_p = np.unravel_index(top_k_p, abs_pcm.shape)
    top_k_s = np.unravel_index(top_k_s, abs_scm.shape)

    return top_k_p, top_k_s