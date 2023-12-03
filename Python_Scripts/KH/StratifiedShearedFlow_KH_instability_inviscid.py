#!/usr/bin/env python
'''
@File    :  StratifiedShearedFlow_KH_instability_inviscid.py
@Time    :  2023/12/03 12:56:13
@Author  :  Daniel Argüeso
@Version :  2.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2022, Daniel Argüeso
@Project :  Master FAMA - Waves and Instability in Geophysical Fluids
@Desc    :  Script to simulate the Kelvin-Helmholtz instability in a stratified sheared flow using dedalus
'''

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import time
import logging

def customize_plots():
    mpl.style.use("seaborn-paper")
    mpl.rcParams["font.size"] = 16
    mpl.rcParams["font.weight"] = "demibold"
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.spines.bottom"] = True
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    
def adjust_spines2(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 5))
        else:
            spine.set_color("none")
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        ax.yaxis.set_ticks([])
    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.xaxis.set_ticks([])
        
        

run_model = True

if run_model:
    root = logging.root
    for h in root.handlers:
        h.setLevel("INFO")

    logger = logging.getLogger(__name__)
    # Set problem domain

    # Aspect ratio 2
    Lx, Ly = (2.0, 1.0)
    nx, ny = (1024, 512)

    # Create bases and domain

    x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=3 / 2)
    y_basis = de.Chebyshev("y", ny, interval=(-Ly / 2, Ly / 2), dealias=3 / 2)
    domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

    # Equations

    problem = de.IVP(domain, variables=["p", "u", "v", "vy", "rho"])

    problem.parameters["g"] = 9.81

    problem.add_equation("dt(u) + dx(p)  = - u*dx(u) - v*dy(u)")
    problem.add_equation("dt(v) + dy(p) + g*rho  = - u*dx(v) - v*vy")
    problem.add_equation("dx(u) + vy = 0")
    problem.add_equation("dt(rho) = -u*dx(rho) - v*dy(rho)")
    problem.add_equation("vy - dy(v) = 0")

    # Boundary conditions

    problem.add_bc("left(v) = 0")
    problem.add_bc("right(v) = 0", condition="(nx != 0)")
    problem.add_bc("integ(p,'y') = 0", condition="(nx == 0)")

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

    a = 0.02
    amp = -0.2
    sigma = 0.2
    flow = -1.0
    N = 4
    u["g"] = flow * np.tanh(4*y / a)
    rho["g"] = amp * np.tanh(3*y / a)
    v["g"] = amp * np.exp(-y**2 / sigma**2) * np.sin(N * np.pi * x / Lx)

    solver.stop_sim_time = 10.01
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf

    initial_dt = 0.2 * Lx / nx
    # cfl = flow_tools.CFL(
    #         solver,
    #         initial_dt=initial_dt,
    #         cadence=10,
    #         safety=0.8,
    #         max_change=1.5,
    #         min_change=0.5,
    #         max_dt=10 * initial_dt,
    #         threshold=0.05,
    #     )
    cfl = flow_tools.CFL(solver, initial_dt, safety=0.8, threshold=0.05)
    cfl.add_velocities(("u", "v"))

    analysis = solver.evaluator.add_file_handler(
        "analysis", sim_dt=0.1, max_writes=10000
    )
    analysis.add_task("rho")
    analysis.add_task("u")
    analysis.add_task("v")
    # Make plot of scalar field
    x = domain.grid(0, scales=domain.dealias)
    y = domain.grid(1, scales=domain.dealias)
    xm, ym = np.meshgrid(x, y)
    fig, axis = plt.subplots(figsize=(8, 5))
    mpl.rcParams["axes.spines.left"] = False
    mpl.rcParams["axes.spines.bottom"] = False
    adjust_spines2(axis, [])
    axis.spines["top"].set_color("none")
    axis.spines["right"].set_color("none")
    axis.spines["left"].set_color("none")
    axis.spines["bottom"].set_linewidth(2)
    rho.set_scales(domain.dealias)
    u.set_scales(domain.dealias)
    v.set_scales(domain.dealias)
    p = axis.pcolormesh(xm, ym, rho["g"].T, cmap="RdBu_r")
    q = axis.quiver(xm[::20,::20],ym[::20,::20], u['g'][::20,::20].T, v['g'][::20,::20].T)
    axis.set_title("Density t = %f" % solver.sim_time)
    axis.set_xlim([0, 2.0])
    axis.set_ylim([-0.5, 0.5])
    logger.info("Starting loop")
    start_time = time.time()
    nt = 0
    while solver.ok:
        dt = cfl.compute_dt()
        solver.step(dt)
        print(solver.iteration)
        if solver.iteration % 10 == 0:
            # Update plot of scalar field
            p.set_array(rho["g"].T)
            q.set_UVC(u['g'][::20,::20].T, v['g'][::20,::20].T)
            axis.set_title("Density t = %f" % solver.sim_time)
            fig.canvas.draw()
            plt.savefig(f"./StratifiedShearedFlow_KH_instability_inviscid_{nt:03d}.png")
            nt += 1


    end_time = time.time()

    # Print statistics
    logger.info("Run time: %f" % (end_time - start_time))
    logger.info("Iterations: %i" % solver.iteration)


# with h5py.File("analysis/analysis_s1/analysis_s1_p0.h5", mode="r") as file:
#     rho = file["tasks"]["rho"]
#     u = file["tasks"]["u"]
#     v = file["tasks"]["v"]
#     t = rho.dims[0]["sim_time"]
#     x = rho.dims[1][0]
#     y = rho.dims[2][0]
#     # Plot data

#     for tstep in range(len(t)):
#         print(tstep)
#         fig, axis = plt.subplots(figsize=(8, 5))
#         p = axis.pcolormesh(x, y, rho[tstep, :, :].T, cmap="RdBu")
#         plt.quiver(x[::10], y[::10], u[tstep, ::10, ::10].T, v[tstep, ::10, ::10].T)
#         axis.set_xlim([0, 2.0])
#         axis.set_ylim([-0.5, 0.5])
#         plt.tight_layout()
#         plt.savefig(f"./StratifiedShearedFlow_KH_instability_inviscid_{tstep:03d}.png")
#         plt.close()
