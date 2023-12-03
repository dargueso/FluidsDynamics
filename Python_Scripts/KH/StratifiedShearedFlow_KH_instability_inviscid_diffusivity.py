import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
import matplotlib.pyplot as plt
import h5py
import time

import logging


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

    problem = de.IVP(domain, variables=["p", "u", "uy", "v", "vy", "rho"])

    problem.parameters["g"] = 9.81
    problem.parameters['Re'] = 2e4

    problem.add_equation("dt(u) + dx(p) - 1/Re*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*dy(u)")
    problem.add_equation("dt(v) + dy(p) - 1/Re*(dx(dx(v)) + dy(vy)) + g*rho = -u*dx(v) - v*vy")
    problem.add_equation("dx(u) + vy = 0")
    problem.add_equation("dt(rho) = -u*dx(rho) - v*dy(rho)")    
    problem.add_equation("vy - dy(v) = 0") 
    problem.add_equation("uy - dy(u) = 0")

    # Boundary conditions
    problem.add_bc("left(u) = 1.0")
    problem.add_bc("right(u) = -1.0")
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
    rho.set_scales(domain.dealias)
    p = axis.pcolormesh(xm, ym, rho["g"].T, cmap="RdBu_r")
    axis.set_title("t = %f" % solver.sim_time)
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
            axis.set_title("t = %f" % solver.sim_time)
            fig.canvas.draw()
            plt.savefig(f"./StratifiedShearedFlow_KH_instability_inviscid_diffusivity_{nt:03d}.png")
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
#         plt.savefig(f"./StratifiedShearedFlow_KH_instability_inviscid_diffusivity_{tstep:03d}.png")
#         plt.close()