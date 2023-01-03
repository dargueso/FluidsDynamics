#!/usr/bin/env python
'''
@File    :  StratifiedShearedFlow_KH_instability.py
@Time    :  2022/12/17 15:39:15
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  Master FAMA - Waves and Instability in Geophysical Fluids
@Desc    :  Script to simulate the Kelvin-Helmholtz instability in a stratified sheared flow using dedalus
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
nx, ny = (1024, 512)

# Create bases and domain

x_basis = de.Fourier('x', nx, interval = (0,Lx), dealias =3/2)
y_basis = de.Chebyshev('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis,y_basis], grid_dtype=np.float64)

#Equations


problem = de.IVP(domain, variables = ['p','u','v','vy','prho'])

problem.parameters['g'] = 9.81

problem.add_equation("dt(u) + dx(p) = - u*dx(u) - v*dy(u)")
problem.add_equation("dt(v) + dy(p) + g*prho = -u*dx(v) - v*vy")
problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(prho) = -u*dx(prho) - v*dy(prho)")
problem.add_equation("vy - dy(v) = 0")

# Boundary conditions


# problem.add_bc("left(u) = 0.5")
# problem.add_bc("right(u) = -0.5")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("integ(p,'y') = 0", condition="(nx == 0)")


#Timestepping

ts = de.timesteppers.RK443

#Initial value problem

solver =  problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
u = solver.state['u']
v = solver.state['v']
vy = solver.state['vy']
p = solver.state['p']
prho = solver.state['prho']



a = 0.05
amp = -0.2
sigma = 0.2
flow = -0.5
u['g'] = flow*np.tanh(y/a)
prho['g'] = -0.1*np.tanh(y/a)
v['g'] = amp*np.exp(-y**2/sigma**2)*np.sin(2*np.pi*x/Lx)

solver.stop_sim_time = 10.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.1*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.5,threshold=0.05)
cfl.add_velocities(('u','v'))

analysis = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=0.1, max_writes=50)
analysis.add_task('prho')
analysis.add_task('u')
analysis.add_task('0.5*(u**2+v**2)',name='KE',scales=(3/2,3/2))
solver.evaluator.vars['Lx'] = Lx

# Make plot of scalar field
x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)
fig, axis = plt.subplots(figsize=(8,5))
prho.set_scales(domain.dealias)
p = axis.pcolormesh(xm, ym, prho['g'].T, cmap='RdBu');
axis.set_title('t = %f' %solver.sim_time)
axis.set_xlim([0,2.])
axis.set_ylim([-0.5,0.5])

logger.info('Starting loop')
start_time = time.time()
nt=0
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)
    if solver.iteration % 10 == 0:
        # Update plot of scalar field
        p.set_array(prho['g'].T)
        axis.set_title('t = %f' %solver.sim_time)
        fig.canvas.draw()
        plt.savefig(f'./KH_instability_dedalus_v2_{nt:03d}.png')
        nt+=1

end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
