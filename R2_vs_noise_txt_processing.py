#%%
# extract info from text

with open('R2_vs_noise_data.txt', 'r') as f:
    data = f.readlines()

data = [line.strip() for line in data]
data = [line.split() for line in data]

# remove empty lists from data
data = [line for line in data if line]

# extract data
noise, R2_values, model_sympy = [], [], []
#%%
for i in range(0, len(data), 2):
    noise.append(float(data[i][3]))
    R2_values.append(float(data[i][-1]))
    model_sympy.append(data[i+1][4:])
# %%
# process model_sympy
# for each model in model_sympy, join the list of strings into one string

model_sympy = [' '.join(model) for model in model_sympy]

# %%
