#!/usr/bin/env python
"""
@File    :  StratifiedShearedFlow_KH_instability.py
@Time    :  2022/12/17 15:39:15
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  Master FAMA - Waves and Instability in Geophysical Fluids
@Desc    :  Script to simulate the Kelvin-Helmholtz instability in a stratified sheared flow using dedalus
"""

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
Lx, Ly = (1.0, 1.0)
nx, ny = (512, 512)

# Create bases and domain

x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=3 / 2)
y_basis = de.Chebyshev("y", ny, interval=(-Ly / 2, Ly / 2), dealias=3 / 2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)


Reynolds = 1200
Schmidt = 9

flow = 10
dens = 0.6
h = 0.25
delta = 0.005
# R = h / delta  # R = 9 Thickness ratio R = h/delta

Jbulk = 0.15
g = 9.81

print(dens * g * h / (flow**2))
# Equations


problem = de.IVP(domain, variables=["p", "u", "uy", "v", "vy", "rho", "rhoy"])

problem.parameters["Re"] = Reynolds
problem.parameters["Sc"] = Schmidt
problem.parameters["J"] = Jbulk
problem.parameters["g"] = 9.81

problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(u) + dx(p) - 1/Re*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*dy(u)")
problem.add_equation(
    "dt(v) + dy(p) - 1/Re*(dx(dx(v)) + dy(vy)) + J*rho = -u*dx(v) - v*vy"
)
problem.add_equation(
    "dt(rho) - 1/(Re*Sc)*(dx(dx(rho)) + dy(rhoy)) = -u*dx(rho) - v*dy(rho)"
)
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("rhoy - dy(rho) = 0")

# Boundary conditions


problem.add_bc("left(uy) = 0")
problem.add_bc("right(uy) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("integ(p,'y') = 0", condition="(nx == 0)")
problem.add_bc("left(rhoy) = 0")
problem.add_bc("right(rhoy) = 0")


# Timestepping

ts = de.timesteppers.RK443

# Initial value problem

solver = problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
u = solver.state["u"]
v = solver.state["v"]
vy = solver.state["vy"]
p = solver.state["p"]
rho = solver.state["rho"]


u_pert = (
    -0.02
    * flow
    * np.cos(2 * np.pi * x / Lx)
    * (1 / np.cosh(2 * y / h))
    * np.tanh(2 * y / h)
)
rho_rand = (
    dens
    / 2
    * (1 - np.abs(np.tanh(2 * y / delta)))
    * np.random.uniform(low=-1, high=1, size=(nx, ny))
)
u_rand = (
    0.05
    * flow
    * (1 - np.abs(np.tanh(2 * y / h)))
    * np.random.uniform(low=-1, high=1, size=(nx, ny))
)
v_rand = (
    0.05
    * flow
    * (1 - np.abs(np.tanh(2 * y / h)))
    * np.random.uniform(low=-1, high=1, size=(nx, ny))
)

u["g"] = flow / 2 * np.tanh(2 * y / h) + u_pert + u_rand
rho["g"] = -dens / 2 * np.tanh(2 * y / delta) + rho_rand
v["g"] = v_rand

solver.stop_sim_time = 10.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.005 * Lx / nx
cfl = flow_tools.CFL(solver, initial_dt, safety=0.5, threshold=0.05)
cfl.add_velocities(("u", "v"))

analysis = solver.evaluator.add_file_handler(
    "analysis_tasks", sim_dt=0.1, max_writes=50
)
analysis.add_task("rho")

# Make plot of scalar field
x = domain.grid(0, scales=domain.dealias)
y = domain.grid(1, scales=domain.dealias)
xm, ym = np.meshgrid(x, y)
fig, axis = plt.subplots(figsize=(8, 5))
rho.set_scales(domain.dealias)
p = axis.pcolormesh(xm, ym, rho["g"].T, cmap="RdBu")
axis.set_title("Density")
axis.set_xlim([0, 1.0])
axis.set_ylim([-0.5, 0.5])

logger.info("Starting loop")
start_time = time.time()
nt = 0
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)
    if solver.iteration % 10 == 0:
        # Update plot of scalar field
        p.set_array(rho["g"].T)
        axis.set_title("Density")
        fig.canvas.draw()
        plt.savefig(f"./Holmboe_instability_dedalus_v2_{nt:03d}_stratified_viscous.png")
        nt += 1


end_time = time.time()

# Print statistics
logger.info("Run time: %f" % (end_time - start_time))
logger.info("Iterations: %i" % solver.iteration)
