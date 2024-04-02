'''
Simulating equations from the paper

We want to implement the equations 1-4 from
https://journals.ametsoc.org/view/journals/atsc/77/1/jas-d-19-0021.1.xml
but without the moisture terms, so, we ignore all the terms involving q, P and E.

The equations are:
dau_u1/dau_t + u1 grad_u1 + f x u1 = -g grad(h1 + h2) - ru1.
dau_u2/dau_t + u2 grad_u2 + f x u2 = -g grad(h1 + sh2).
dau_h1/dau_t + grad(h1u1) = (h1^f(y) - h1)/tau_h.
daus_h2/dau_t + grad(h2u2) = (h2^f(y) - h2)/tau_h.
'''

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600

# Parameters
Nphi = 256
Ntheta = 128
dealias = 3/2
R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e5 * meter**2 / second / 32**2 # Hyperdiffusion matched at ell=32
g = 9.80616 * meter / second**2
H = 1e4 * meter
timestep = 600 * second
stop_sim_time = 360 * hour
dtype = np.float64


# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
h = dist.Field(name='h', bases=basis)

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))

# Initial conditions: zonal jet
phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi
umax = 80 * meter / second
lat0 = np.pi / 7
lat1 = np.pi / 2 - lat0
en = np.exp(-4 / (lat1 - lat0)**2)
jet = (lat0 <= lat) * (lat <= lat1)
u_jet = umax / en * np.exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))
u['g'][0][jet]  = u_jet

# Initial conditions: balanced height
c = dist.Field(name='c')
problem = d3.LBVP([h, c], namespace=locals())
problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
problem.add_equation("ave(h) = 0")
solver = problem.build_solver()
solver.solve()

# Initial conditions: perturbation
lat2 = np.pi / 4
hpert = 120 * meter
alpha = 1 / 3
beta = 1 / 15
h['g'] += hpert * np.cos(lat) * np.exp(-(phi/alpha)**2) * np.exp(-((lat2-lat)/beta)**2)
