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
            Non-dimensional equations used in this code: Singh and Srinivasan (2014)
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


Pr = 7
Sc = 700
Rat = 3.33e9
Rrho0 = 6

#Equations

problem = de.IVP(domain, variables = ['p','u','uy','v','vy','T','Ty','S','Sy'])

problem.parameters['Pr'] = Pr
problem.parameters['Sc'] = Sc
problem.parameters['Rat'] = Rat
problem.parameters['Rrho0'] = Rrho0


problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(u) + dx(p) - Pr*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy")
problem.add_equation("dt(v) + dy(p) - Pr*(dx(dx(v)) - dy(vy)) + Pr*Rat*(S/Rrho0 - T) =  - u*dx(v) - v*vy")
problem.add_equation("dt(T) - (dx(dx(T)) + dy(Ty)) = - u*dx(T) - v*Ty")
problem.add_equation("dt(S) - (Pr/Sc)*(dx(dx(S)) + dy(Sy)) = - u*dx(S) - v*Sy")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("Ty - dy(T) = 0")
problem.add_equation("Sy  - dy(S) = 0")





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
S = solver.state['S']
Ty = solver.state['Ty']
Sy = solver.state['Sy']

gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

a = 0.02
u['g'] = 0
v['g'] = 0
T['g'] = 0.5*np.tanh(y/a)+0.5
S['g'] = 0.5*np.tanh(y/a)+0.5


xloc = 128
Tloc = T['g'][xloc,:]
Sloc = S['g'][xloc,:]


fig2, axis2 = plt.subplots(figsize=(8,5))
line1 = axis2.plot(Tloc,np.arange(len(Tloc)))
line2 = axis2.plot(Sloc, np.arange(len(Sloc)))
fig2.savefig(f'./TProfile_Double-diffusive_instability_000.png')


initial_dt = 0.05*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.5,threshold=0.05)
cfl.add_velocities(('u','v'))

# Make plot of scalar field
x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)

T.set_scales(domain.dealias)
S.set_scales(domain.dealias)
u.set_scales(domain.dealias)
v.set_scales(domain.dealias)
vel = (u['g']**2+v['g']**2)**0.5

fig, axis = plt.subplots(1,3,figsize=(80,20))

p1 = axis[0].pcolormesh(xm, ym, v['g'].T, cmap='PuOr')
axis[0].set_title(f'Vert veloc. (t = {solver.sim_time:2.2f})')

p2 = axis[1].pcolormesh(xm, ym, T['g'].T, cmap='RdYlBu_r')
axis[1].set_title(f'Temperature (t = {solver.sim_time:2.2f})')

p3 = axis[2].pcolormesh(xm, ym, S['g'].T, cmap='bwr')
axis[2].set_title(f'Salinity (t = {solver.sim_time:2.2f})')

fig.savefig(f'./Double-diffusive_instability_000.png')

logger.info('Starting loop')
start_time = time.time()
nt=1
while solver.ok:
    dt = cfl.compute_dt()
    dt = 0.05*Lx/nx
    solver.step(dt)
    
    if solver.iteration % 10 == 0:
        # Update plot of scalar field
        vel = (u['g']**2+v['g']**2)**0.5
        rho = Pr*Rat*(S['g']/Rrho0 - T['g'])
        p1.set_array(np.ravel(v['g'].T))
        p2.set_array(np.ravel(T['g'].T))
        p3.set_array(np.ravel(S['g'].T))

        axis[0].set_title(f'Vert veloc. (t = {solver.sim_time:2.2f})')
        axis[1].set_title(f'Temperature  (t = {solver.sim_time:2.2f})')
        axis[2].set_title(f'Salinity (t = {solver.sim_time:2.2f})')

        fig.canvas.draw()
        fig.savefig(f'./Double-diffusive_instability_{nt:03d}.png')

        Tloc = T['g'][xloc,:]
        Sloc = S['g'][xloc,:]

        fig2, axis2 = plt.subplots(figsize=(8,5))
        line1 = axis2.plot(Tloc,np.arange(len(Tloc)))
        line2 =  axis2.plot(Sloc, np.arange(len(Sloc)))
        fig2.savefig(f'./TProfile_Double-diffusive_instability_{nt:03d}.png')


        nt+=1




end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
