'''
Simulating equations from the paper

We want to implement the equations 1-4 from
https://journals.ametsoc.org/view/journals/atsc/77/1/jas-d-19-0021.1.xml
but without the moisture terms, so, we ignore all the terms involving q, P and E.

The equations are:
dau_u1/dau_t + u1 grad_u1 + f x u1 = -g grad(h1 + h2) - ru1.
dau_u2/dau_t + u2 grad_u2 + f x u2 = -g grad(h1 + sh2).
dau_h1/dau_t + grad(h1u1) = (h1f(y) - h1)/tau_h.
dau_h2/dau_t + grad(h2u2) = (h2f(y) - h2)/tau_h.
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
u1 = dist.VectorField(coords, name='u1', bases=basis)
u2 = dist.VectorField(coords, name='u2', bases=basis)
h1 = dist.Field(name='h1', bases=basis)
h2 = dist.Field(name='h2', bases=basis)

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
u1['g'][0][jet]  = u_jet
u2['g'][0][jet]  = u_jet

# Initial conditions: balanced height
c1 = dist.Field(name='c1')
problem = d3.LBVP([h1, c1], namespace=locals())
problem.add_equation("g*lap(h1) + c1 = - div(u@grad(u) + 2*Omega*zcross(u))")
problem.add_equation("ave(h1) = 0")
solver = problem.build_solver()
solver.solve()

c2 = dist.Field(name='c2')
problem = d3.LBVP([h2, c2], namespace=locals())
problem.add_equation("g*lap(h2) + c2 = - div(u@grad(u) + 2*Omega*zcross(u))")
problem.add_equation("ave(h2) = 0")
solver = problem.build_solver()
solver.solve()

# Initial conditions: perturbation
lat2 = np.pi / 4
hpert = 120 * meter
alpha = 1 / 3
beta = 1 / 15
h1['g'] += hpert * np.cos(lat) * np.exp(-(phi/alpha)**2) * np.exp(-((lat2-lat)/beta)**2)
h2['g'] += hpert * np.cos(lat) * np.exp(-(phi/alpha)**2) * np.exp(-((lat2-lat)/beta)**2)

# Problem
problem = d3.IVP([u1, h1, u2, h2], namespace=locals())
problem.add_equation("dt(u1) + u1*grad(u1) + f*u1 = -g*grad(h1 + h2) - r*u1")
problem.add_equation("dt(u2) + u2*grad(u2) + f*u2 = -g*grad(h1 + s*h2)")
problem.add_equation("dt(h1) + grad(h1*u1) = (h1f(y) - h1)/tau_h")
problem.add_equation("dt(h2) + grad(h2*u2) = (h2f(y) - h2)/tau_h")

# Figure out how to encode the variables: s, r, f, g, h1f, h2f and tau_h

# Solver

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=1*hour, max_writes=10)
snapshots.add_task(h1, name='height_1')
snapshots.add_task(h2, name='height_2')
snapshots.add_task(u1, name='velocity_1')
snapshots.add_task(u2, name='velocity_2')
snapshots.add_task(-d3.div(d3.skew(u1)), name='vorticity_1')
snapshots.add_task(-d3.div(d3.skew(u2)), name='vorticity_2')
snapshots.add_task(d3.lap(h1), name='lap_of_height_1')
snapshots.add_task(d3.lap(h2), name='lap_of_height_2')
snapshots.add_task(d3.lap(d3.lap(h1)), name='lap_of_lap_of_height_1')
snapshots.add_task(d3.lap(d3.lap(h2)), name='lap_of_lap_of_height_2')
snapshots.add_task(d3.lap(d3.lap(u1)), name='lap_of_lap_of_velocity_1')
snapshots.add_task(d3.lap(d3.lap(u2)), name='lap_of_lap_of_velocity_2')
snapshots.add_task(d3.div(u1), name='div_of_velocity_1')
snapshots.add_task(d3.div(u2), name='div_of_velocity_2')
snapshots.add_task(d3.div(h1*u1), name='div_of_h_times_u1')
snapshots.add_task(d3.div(h2*u2), name='div_of_h_times_u2')
snapshots.add_task(d3.grad(h1), name='grad_of_height_1')
snapshots.add_task(d3.grad(h2), name='grad_of_height_2')
snapshots.add_task(d3.grad(u1), name='grad_of_velocity_1')
snapshots.add_task(d3.grad(u2), name='grad_of_velocity_2')
snapshots.add_task(zcross(u1), name='zcross_of_velocity_1')
snapshots.add_task(zcross(u2), name='zcross_of_velocity_2')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration,
                                                         solver.sim_time,
                                                         timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
