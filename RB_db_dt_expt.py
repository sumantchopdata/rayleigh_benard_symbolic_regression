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

    # Now fit the model on the original data without noise
    model.fit(X_noisy, y)

    print('For noise =', noise, 'the best model R2 score without noise in y is ',
            model.score(X_noisy, y))

    del model, X_noisy, y_noisy
#%%
print(R2_noisy_list)

# This is the output of the above code
R2_noisy_list = [0.839254781420806, 0.8388748587271538, 0.8379228016990612,
 0.836609871349761, 0.8347133487306104, 0.8325543873346839, 0.8239453351249085,
 0.8120196958818136, 0.7973017168851783, 0.5586829967745214, 0.5530145825589219,
 0.5439517971764247, 0.5211959635976793, 0.49527850119046013, 0.4624842910226856,
 0.42827420271330263, 0.39447706057001997, 0.357843451433676, 0.2955140782824469,
 0.2406346545487683, 0.19446241393600627, 0.15618477035564193, 0.1233695336913131,
 0.07498618311523109, 0.04303638760679829, 0.029571994901882337, 0.018556640915129874,
 0.014051161044079219, 0.00860789409242746, 0.006225562333402768, 0.004330694444944161,
 0.0025949855949624068, 0.001489100410956512, 0.000797177493995771, 0.0007319860241528087]

print(model_sympys_list)

# This is the output of the above code
model_sympys_list = ['0.00069614046*x0 - x2', '0.000699487*x0 - x2',
 '0.00070116983*x0 - x2', '0.0006990202*x0 - x2', '0.0006991287*x0 - x2',
 '0.0006946154*x0 - x2', '0.0006969347*x0 - x2', '0.0006880454*x0 - x2',
 '0.0006931664*x0 - x2', '-0.68597394*x2', '-0.66939783*x2', '-0.6526355*x2',
 '-0.6160177*x2', '-0.5836424*x2', '-0.5464623*x2', '-0.51449287*x2', '-0.4864223*x2',
 '-0.45669678*x2', '-0.4076762*x2', '-0.3640049*x2', '-0.32944205*x2', '-0.2928266*x2',
 '-0.2624302*x2', '-0.20802976*x2', '-0.16211073*x2', '-0.13912684*x2', '-0.10783761*x2',
 '-0.10044246*x2', '-0.0801621*x2', '-0.06858837*x2', '-0.06014602*x2', '-0.04748086*x2',
 '-0.03571533*x2', '-0.026206696*x2', '-0.022418577*x2']
#%%    
# plot the R2 scores for different values of noise

# Create the signal-to-noise ratio list
snr = [np.mean(np.abs(X))/noise for noise in noise_list]

snr = snr[1:]
R2_noisy_list_without_inf = R2_noisy_list[1:]

plt.scatter(snr, R2_noisy_list_without_inf)
plt.plot(snr, R2_noisy_list_without_inf)

plt.xlabel('mean(abs(X))/noise  for different noise values')
plt.xscale('log')
plt.ylabel('R2 Score')
plt.ylim((-0.1,1))

plt.text(snr[7]-700, R2_noisy_list[8]+0.04, str(
                                (round(snr[8],2), round(R2_noisy_list[8],2))))
plt.text(snr[8]+150, R2_noisy_list[9], str(
                                (round(snr[9],2), round(R2_noisy_list[9],2))))

plt.text(snr[1], R2_noisy_list[1]-0.08, str(
                                (round(snr[1],2), round(R2_noisy_list[1],2))))
plt.text(snr[-1]-5, R2_noisy_list[-1]-0.05, str(
                                (round(snr[-1],2), round(R2_noisy_list[-1],2))))

plt.title('R2 score vs Signal-to-Noise Ratio for db/dt model')
plt.savefig('R2_vs_snr_for_db_dt.png', dpi=300)
plt.show()

#%%
plt.scatter(noise_list[1:], R2_noisy_list[1:])
plt.plot(noise_list[1:], R2_noisy_list[1:])

plt.xlabel('Noise')
plt.xscale('log')
plt.ylabel('R2 Score')
plt.ylim((-0.1,1))

plt.text(noise_list[8], R2_noisy_list[8]+0.035, str(
                        (round(noise_list[8],4), round(R2_noisy_list[8],3))))
plt.text(noise_list[9]-0.0115, R2_noisy_list[9]+0.02, str(
                        (round(noise_list[9],4), round(R2_noisy_list[9],3))))

plt.text(noise_list[1]-0.0001, R2_noisy_list[1]-0.08, str(
                        (round(noise_list[1],4), round(R2_noisy_list[1],3))))
plt.text(noise_list[-1]-0.25, R2_noisy_list[-1]-0.05, str(
                        (round(noise_list[-1],4), round(R2_noisy_list[-1],3))))

plt.title('R2 score vs Noise for db/dt model')
plt.savefig('R2_vs_noise_for_db_dt.png', dpi=300)
plt.show()