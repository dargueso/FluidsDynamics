#!/usr/bin/env python
'''
@File    :  DoubleDiffusive_saltfingers_instability.py
@Time    :  2023/01/22 21:44:22
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2023, Daniel Argüeso
@Project :  Master FAMA - Waves and Instability in Geophysical Fluids
@Desc    :  Script to simulate the salt fingers (double-diffusive) instability using dedalus
            It is described here: https://www.youtube.com/watch?v=3GrvRaztgRA
            and the equations can also be found in the book "Instabilities in Geophysical Flows" by Smyth and Carpernter
'''

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
import matplotlib.pyplot as plt
import h5py
import time

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
    
logger = logging.getLogger(__name__)
# Set problem domain

# Aspect ratio 2
Lx, Ly = (2., 1.)
nx, ny = (512, 256)

# Create bases and domain

x_basis = de.Fourier('x', nx, interval = (0,Lx), dealias =3/2)
y_basis = de.Chebyshev('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis,y_basis], grid_dtype=np.float64)


# Problem
problem = de.IVP(domain, variables=  ['psi','psiy','T','S'])

# Substitutions

problem.substitutions["J(A,B)"] = "  dx(A)*dy(B) - dy(A)*dx(B) "
problem.substitutions["L(A)"] = "  d(A,x=2) + d(A,y=2) "

problem.substitutions["zeta"] = "  d(psi,x=2) + dy(psiy) "
problem.substitutions["u"] = " -psiy "
problem.substitutions["v"] = "  dx(psi) "

problem.parameters["tau"] = 0.1
problem.parameters["Sc"] = 100
problem.parameters["Rrho"] = 1


problem.add_equation("dt(T) + dx(psi) = 1/tau * L(T) - J(psi,T)")
problem.add_equation("dt(S) + dx(psi) = 1/tau * L(S) - J(psi,S)")
problem.add_equation("1/Sc * dt(zeta) - 1/tau * dx(T) + 1/(tau*Rrho) * dx(S) = L(zeta)  - 1/Sc * J(psi,zeta)")
problem.add_equation("psiy - dy(psi) = 0")



# Boundary conditions

#problem.add_bc("left(S) = 1.1")
#problem.add_bc("right(S) = 0.9")
# problem.add_bc("left(T) = 300")
# problem.add_bc("right(T) = 298")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0")


#Timestepping

ts = de.timesteppers.RK443

#Initial value problem

solver = problem.build_solver(ts)
solver.stop_sim_time = 10.1
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

x = domain.grid(0)
y = domain.grid(1)
T = solver.state['T']
S = solver.state['S']
psi = solver.state['psi']
psiy = solver.state['psiy']

T["g"] = 295+(305-295)*y
S["g"] = 1+(0-1)*y


psi['g'] = 0
psi.differentiate('y',out=psiy)

initial_dt = 0.2*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
cfl.add_velocities(('u','v'))

analysis = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=0.1, max_writes=50)
analysis.add_task('T')
analysis.add_task('S')

# Make plot of scalar field
x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)
fig, axis = plt.subplots(figsize=(10,5))
T.set_scales(domain.dealias)
p = axis.pcolormesh(xm, ym, T['g'].T, cmap='RdBu_r')
axis.set_title(f'Temperature (t = {solver.sim_time:2.2f})')
plt.savefig(f'./Double-diffusive_instability_dedalus_v2_init.png')
logger.info('Starting loop')
start_time = time.time()
nt=0
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)
    if solver.iteration % 1 == 0:
        # Update plot of scalar field
        p.set_array(np.ravel(T['g'].T))
        axis.set_title(f'Temperature (t = {solver.sim_time:2.2f})')
        fig.canvas.draw()
        plt.savefig(f'./Double-diffusive_instability_dedalus_v2_{nt:03d}.png')
        nt+=1

end_time = time.time()


# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
