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
            Equations used: https://gfd.whoi.edu/wp-content/uploads/sites/18/2018/03/lecture10_136325.pdf
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
Lx, Ly = (1., 2.)
nx, ny = (256, 512)

# Create bases and domain

x_basis = de.Fourier('x', nx, interval = (0,Lx), dealias =3/2)
y_basis = de.Chebyshev('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis,y_basis], grid_dtype=np.float64)


nu = 1e-2
alpha = 1e-1
beta = 1e-2
Kt = 1
Ks = 1e-2
g = 9.81

#Equations


problem = de.IVP(domain, variables = ['p','u','uy','v','vy','T','Ty','S','Sy'])

problem.parameters['g'] = g
problem.parameters['alpha'] = alpha
problem.parameters['beta'] = beta
problem.parameters['Kt'] = Kt
problem.parameters['Ks'] = Ks
problem.parameters['nu'] = nu
problem.parameters['Tmeany'] = 100
problem.parameters['Smeany'] = 100

problem.add_equation("dt(u) + dx(p) - nu*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy")
problem.add_equation("dt(v) + dy(p) - nu*(dx(dx(v)) + dy(vy)) = - u*dx(v) - v*vy - g*(beta*S-T*alpha)")
problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(T) - Kt*(dx(dx(T)) + dy(Ty)) =  - v*Tmeany")
problem.add_equation("dt(S) - Ks*(dx(dx(S)) + dy(Sy)) =  - v*Smeany")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("Ty - dy(T) = 0")
problem.add_equation("Sy - dy(S) = 0")

# Boundary conditions

problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("integ(p,'y') = 0", condition="(nx == 0)")
problem.add_bc("left(Ty) = 0")
problem.add_bc("right(Ty) = 0")
problem.add_bc("left(Sy) = 0")
problem.add_bc("right(Sy) = 0")


#Timestepping

ts = de.timesteppers.RK443

#Initial value problem

solver = problem.build_solver(ts)
solver.stop_sim_time = 1000.1
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

x = domain.grid(0)
y = domain.grid(1)
u = solver.state['u']
u = solver.state['uy']
v = solver.state['v']
vy = solver.state['vy']
p = solver.state['p']
T = solver.state['T']
Ty = solver.state['Ty']
S = solver.state['S']
Sy = solver.state['Sy']

gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]
noise1 = rand.standard_normal(gshape)[slices]
noise2 = rand.standard_normal(gshape)[slices]
a = 0.02
amp = -0.2
sigma = 0.2
flow = -1.0
u['g'] = 0
#u['g'] = flow*np.tanh(y/a)
T['g'] = 0
S['g'] = 0
v['g'] = noise2/400


fig2, axis2 = plt.subplots(figsize=(8,5))
line1 = axis2.plot(T['g'][128,:],np.arange(len(T['g'][128,:])))
line2 = axis2.plot(beta*S['g'][128,:] - alpha*T['g'][128,:],np.arange(len(T['g'][128,:])),'r')
line3 = axis2.plot(S['g'][128,:],np.arange(len(S['g'][128,:])))
fig2.savefig(f'./TProfile_Double-diffusive_instability_000.png')


initial_dt = 0.05*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.5,threshold=0.05)
cfl.add_velocities(('u','v'))

analysis = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=0.1, max_writes=50)
analysis.add_task('T')
analysis.add_task('u')
analysis.add_task('v')
solver.evaluator.vars['Lx'] = Lx

# Make plot of scalar field
x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)
fig, axis = plt.subplots(1,4,figsize=(40,20))
T.set_scales(domain.dealias)
S.set_scales(domain.dealias)
u.set_scales(domain.dealias)
v.set_scales(domain.dealias)
vel = (u['g']**2+v['g']**2)**0.5
rho = beta*S['g'] - alpha*T['g']
p0 = axis[0].pcolormesh(xm, ym, S['g'].T, cmap='bwr')
axis[0].set_title(f'Salinity (t = {solver.sim_time:2.2f})')

p1 = axis[1].pcolormesh(xm, ym, T['g'].T, cmap='RdYlBu_r')
axis[1].set_title(f'Temperature (t = {solver.sim_time:2.2f})')

p2 = axis[2].pcolormesh(xm, ym, rho.T, cmap='RdBu')
axis[2].set_title(f'Density (t = {solver.sim_time:2.2f})')

p3 = axis[3].pcolormesh(xm, ym, v['g'].T, cmap='PuOr')
axis[3].set_title(f'Vert veloc. (t = {solver.sim_time:2.2f})')

fig.savefig(f'./Double-diffusive_instability_000.png')





logger.info('Starting loop')
start_time = time.time()
nt=1
while solver.ok:
    dt = cfl.compute_dt()
    #dt = 0.05*Lx/nx
    solver.step(dt)
    
    if solver.iteration % 10 == 0:
        # Update plot of scalar field
        vel = (u['g']**2+v['g']**2)**0.5
        rho = beta*S['g'] - alpha*T['g']
        p0.set_array(np.ravel(S['g'].T)) 
        p1.set_array(np.ravel(T['g'].T))
        p2.set_array(np.ravel(rho.T))
        p3.set_array(np.ravel(v['g'].T))
        axis[0].set_title(f'Salinity (t = {solver.sim_time:2.2f})')
        axis[1].set_title(f'Temperature  (t = {solver.sim_time:2.2f})')
        axis[2].set_title(f'Density (t = {solver.sim_time:2.2f})')
        axis[3].set_title(f'Vert veloc. (t = {solver.sim_time:2.2f})')
        
        fig.canvas.draw()
        fig.savefig(f'./Double-diffusive_instability_{nt:03d}.png')

        fig2, axis2 = plt.subplots(figsize=(8,5))
        line1 = axis2.plot(T['g'][128,:],np.arange(len(T['g'][128,:])))
        line2 = axis2.plot(beta*S['g'][128,:] - alpha*T['g'][128,:],np.arange(len(T['g'][128,:])),'r')
        line3 = axis2.plot(S['g'][128,:],np.arange(len(S['g'][128,:])))
        fig2.savefig(f'./TProfile_Double-diffusive_instability_{nt:03d}.png')
        nt+=1




end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
