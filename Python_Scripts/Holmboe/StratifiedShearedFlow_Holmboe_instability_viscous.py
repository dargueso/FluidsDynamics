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
nx, ny = (256, 128)

# Create bases and domain

x_basis = de.Fourier('x', nx, interval = (0,Lx), dealias =3/2)
y_basis = de.Chebyshev('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis,y_basis], grid_dtype=np.float64)


Reynolds = 3e5
Schmidt = 1.0


#Equations


problem = de.IVP(domain, variables=  ['p','u','v','uy','vy','S','Sy'])

problem.parameters['Re'] = Reynolds
problem.parameters['Sc'] = Schmidt
problem.parameters['g'] = 9.81

problem.add_equation("dt(u) + dx(p) - 1/Re*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy")
problem.add_equation("dt(v) + dy(p) - 1/Re*(dx(dx(v)) + dy(vy)) - S = - u*dx(v) - v*vy")
problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(S) - 1/(Re*Sc)*(dx(dx(S)) + dy(Sy)) = - u*dx(S) - v*Sy")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("Sy - dy(S) = 0")


# Boundary conditions

problem.add_bc("left(u) = 1.0")
problem.add_bc("right(u) = -1.0")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("left(p) = 0", condition="(nx == 0)")
problem.add_bc("left(S) = 0")
problem.add_bc("right(S) = 1")


#Timestepping

ts = de.timesteppers.RK443

#Initial value problem

solver =  problem.build_solver(ts)


x = domain.grid(0)
y = domain.grid(1)
u = solver.state['u']
uy = solver.state['uy']
v = solver.state['v']
vy = solver.state['vy']
S = solver.state['S']
Sy = solver.state['Sy']


a = 0.02
sigma = 0.2
flow = 5.0
amp = -0.1
u['g'] = flow*np.tanh(y/(4*a))
#v['g'] = amp*np.sin(2.0*np.pi*x/Lx)*np.exp(-(y*y)/(sigma*sigma))
v['g'] = amp*np.exp(-y**2/sigma**2)*np.sin(2*np.pi*x/Lx)
S['g'] = 1.0*(1+np.tanh(2*y/a))
#S['g'] = 1/(1-0.001*np.tanh(5*y))
u.differentiate('y',out=uy)
v.differentiate('y',out=vy)
S.differentiate('y',out=Sy)

solver.stop_sim_time = 10.1
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.2*Lx/nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
cfl.add_velocities(('u','v'))

analysis = solver.evaluator.add_file_handler('analysis_tasks', sim_dt=0.1, max_writes=50)
analysis.add_task('S')
analysis.add_task('u')
solver.evaluator.vars['Lx'] = Lx
#analysis.add_task("integ(S,'x')/Lx", name='S profile')


# Make plot of scalar field
x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)
fig, axis = plt.subplots(figsize=(10,5))
p = axis.pcolormesh(xm, ym, S['g'].T, cmap='RdBu_r');
axis.set_title(f'Buoyancy (t = {solver.sim_time:2.2f})')
axis.set_xlim([0,2.])
axis.set_ylim([-0.5,0.5])
plt.savefig(f'./Holmboe_instability_dedalus_v2_000_stratified_viscous.png')

logger.info('Starting loop')
start_time = time.time()
nt=1
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)
    if solver.iteration % 10 == 0:
        # Update plot of scalar field
        p.set_array(np.ravel(S['g'].T))
        axis.set_title(f'Buoyancy (t = {solver.sim_time:2.2f})')
        fig.canvas.draw()
        plt.savefig(f'./Holmboe_instability_dedalus_v2_{nt:03d}_stratified_viscous.png')
        nt+=1

end_time = time.time()


# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
