#!/usr/bin/env python
"""
@File    :  KHInstability_dedalus.py
@Time    :  2022/12/02 17:21:17
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  None
@Desc    :  None
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import animation
from dedalus import public as de
from dedalus.extras import flow_tools

#####################################################################
#####################################################################
import logging

root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)

#####################################################################
#####################################################################


# Aspect ratio 2
Lx, Ly = (2.0, 1.0)
nx, ny = (256, 128)

# Create bases and domain
x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=3 / 2)
y_basis = de.Chebyshev("y", ny, interval=(-Ly / 2, Ly / 2), dealias=3 / 2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

#####################################################################
#####################################################################
# Equations
problem = de.IVP(domain, variables=["p", "u", "v", "uy", "vy", "s", "Sy"])

Reynolds = 1e4
Schmidt = 1.0
Prandtl = 1.0

problem.parameters["Re"] = Reynolds
problem.parameters["Sc"] = Schmidt
problem.parameters["Pr"] = Prandtl

problem.add_equation("dt(u) + dx(p) - 1/Re*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy")
problem.add_equation("dt(v) + dy(p) - 1/Re*(dx(dx(v)) + dy(vy)) = - u*dx(v) - v*vy")
problem.add_equation("dx(u) + vy = 0")

problem.add_equation("dt(s) - 1/(Re*Sc)*(dx(dx(s)) + dy(Sy)) = - u*dx(s) - v*Sy")

problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("Sy - dy(s) = 0")


#####################################################################
#####################################################################
# Boundary conditions
problem.add_bc("left(u) = 0.5")
problem.add_bc("right(u) = -0.5")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("integ(p,'y') = 0", condition="(nx == 0)")
problem.add_bc("left(s) = 0")
problem.add_bc("right(s) = 1")

#####################################################################
#####################################################################
# timesteeping

ts = de.timesteppers.RK443

#####################################################################
#####################################################################
# Initial value problem

solver = problem.build_solver(ts)


x = domain.grid(0)
y = domain.grid(1)
u = solver.state["u"]
uy = solver.state["uy"]
v = solver.state["v"]
vy = solver.state["vy"]
p = solver.state["p"]
s = solver.state["s"]
sy = solver.state["Sy"]

a = 0.05
sigma = 0.2
flow = -1.0
u["g"] = flow * np.tanh(y / a)
s["g"] = 0.5 * (1 + np.tanh(y / a))

amp = -0.5
sigma = 0.2
v["g"] = amp * np.exp(-(y**2) / sigma**2) * np.sin(8 * np.pi * x / Lx)


plt.close()
plt.plot(v["g"][int(nx / 4), :], y[0])
plt.ylabel("y")
plt.xlabel("vertical velocity")
plt.savefig("KH_velocity.png")

#####################################################################
#####################################################################

solver.stop_sim_time = 6.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.2 * Lx / nx
cfl = flow_tools.CFL(solver, initial_dt, safety=0.8, threshold=0.05)
cfl.add_velocities(("u", "v"))

#####################################################################
#####################################################################

# Analysis

analysis = solver.evaluator.add_file_handler(
    "analysis_tasks", sim_dt=0.1, max_writes=50
)
analysis.add_task("s")
analysis.add_task("u")
analysis.add_task("0.5*(u**2+v**2)", name="KE", scales=(3 / 2, 3 / 2))
analysis.add_task("0.5*(dx(v)-uy)**2", name="enstrophy")
solver.evaluator.vars["Lx"] = Lx
analysis.add_task("integ(s,'x')/Lx", name="s profile")

#####################################################################
#####################################################################

# Main loop

# Make plot of scalar field
x = domain.grid(0, scales=domain.dealias)
y = domain.grid(1, scales=domain.dealias)
xm, ym = np.meshgrid(x, y)
fig, axis = plt.subplots(figsize=(8, 5))
s.set_scales(domain.dealias)
p = axis.pcolormesh(xm, ym, s["g"].T, cmap="RdBu_r")


def init():
    axis.set_xlim([0, 2.0])
    axis.set_ylim([-0.5, 0.5])
    return (p,)


def update(frame):

    if solver.ok:
        dt = cfl.compute_dt()
        solver.step(dt)
        p.set_array(np.ravel(s["g"][:-1, :-1].T))
        axis.set_title(f"t = {solver.sim_time}")


# mimation = FuncAnimation(fig, update,interval = 300)

anim = animation.FuncAnimation(fig, update, frames=300, interval=300)
anim.save("ShearedFlow_KH_instability_viscous.mp4", fps=20)
