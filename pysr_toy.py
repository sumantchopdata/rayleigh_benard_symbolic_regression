#%%
# PySR Toy example from the documentation
import numpy as np
from pysr import *

X = 2 * np.random.randn(100, 5)
y = 2 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2
model = PySRRegressor(binary_operators=["+", "-", "*", "/"])
model.fit(X, y)
print(model.equations_)
#%%
X = 2 * np.random.randn(100, 5)
y = 1 / X[:, 0]
model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1/x},
)
model.fit(X, y)
print(model.equations_)
#%%
X = 2 * np.random.randn(100, 5)
y = 1 / X[:, [0, 1, 2]]
model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1/x},
)
model.fit(X, y)
# %%
model.latex()[0]
# %%
import matplotlib.pyplot as plt
plt.scatter(y[:, 0], model.predict(X)[:, 0])
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()
# %%
X = np.random.randn(300, 30)
y = X[:, 3]**2 - X[:, 19]**2 + 1.5

model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp"],
    # select_k_features=5,
)

model.fit(X, y)
# %%
X = np.random.randn(100, 5)
noise = np.random.randn(100) * 0.1
y = np.exp(X[:, 0]) + X[:, 1] + X[:, 2] + noise

model_1 = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp"],
    denoise=False,
)

model_1.fit(X, y)
print(model_1.equations_)
#%%
model_2 = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp"],
    denoise=True,
)

model_2.fit(X, y)
print(model_2.equations_)
#%%
import numpy as np

X = np.random.randn(100, 1) + 1j * np.random.randn(100, 1)
y = (1 + 2j) * np.cos(X[:, 0] * (0.5 - 0.2j))

model = PySRRegressor(
    binary_operators=["+", "-", "*"], unary_operators=["cos"], niterations=100,
)

model.fit(X, y)
print(model.equations_)