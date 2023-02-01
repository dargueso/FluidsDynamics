#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""beta plane barotropic vorticity model.

Solve the barotropic vorticity equation in two dimensions

    D/Dt[ω] = 0                                                             (1)

where ω = ζ + f is absolute vorticity.  ζ is local vorticity ∇ × u and
f is global rotation.

Assuming an incompressible two-dimensional flow u = (u, v),
the streamfunction ψ = ∇ × (ψ êz) can be used to give (u,v)

    u = -∂/∂y[ψ]         v = ∂/∂x[ψ]                                        (2)

and therefore local vorticity is given by the Poisson equation

    ζ = ∆ψ                                                                  (3)

Since ∂/∂t[f] = 0 equation (1) can be written in terms of the local vorticity

        D/Dt[ζ] + u·∇f = 0
    =>  D/Dt[ζ] = -vβ                                                       (4)

using the beta-plane approximation f = f0 + βy.  This can be written entirely
in terms of the streamfunction and this is the form that will be solved
numerically.

    D/Dt[∆ψ] = -β ∂/∂x[ψ]                                                   (5)

"""
import logging
import h5py
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import flow_tools


run_model = False

if run_model:
    root = logging.root
    for h in root.handlers:
        h.setLevel("INFO")

    logger = logging.getLogger(__name__)

    N = 256
    Lx, Ly = (3.0, 1.0)
    nx, ny = (N, N)
    beta = 15.0

    # setup the domain
    x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=3 / 2)
    y_basis = de.Fourier("y", ny, interval=(0, Ly), dealias=3 / 2)
    domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

    problem = de.IVP(domain, variables=["psi"])

    # solve the problem from the equations
    # ζ = Δψ
    # ∂/∂t[∆ψ] + β ∂/∂x[ψ] = -J(ζ, ψ)

    # Everytime you ask for one of the expression on the left, you will get the expression on the right.
    problem.substitutions["zeta"] = "  d(psi,x=2) + d(psi,y=2) "
    problem.substitutions["u"] = " -dy(psi) "
    problem.substitutions["v"] = "  dx(psi) "

    # This pattern matches for the 'thing' arguements. They don't have to be called 'thing'.
    problem.substitutions["L(thing_1)"] = "  d(thing_1,x=2) + d(thing_1,y=2) "
    problem.substitutions[
        "J(thing_1,thing_2)"
    ] = "  dx(thing_1)*dy(thing_2) - dy(thing_1)*dx(thing_2) "

    # You can combine things if you want
    problem.substitutions["HD(thing_1)"] = "  -D*L(L(thing_1)) "

    problem.parameters["beta"] = beta
    problem.parameters["D"] = 0.01  # hyperdiffusion coefficient

    problem.add_equation(
        "dt(zeta) + beta*v  = J(psi,zeta) ", condition="(nx != 0) or  (ny != 0)"
    )
    problem.add_equation("psi = 0", condition="(nx == 0) and (ny == 0)")

    solver = problem.build_solver(de.timesteppers.CNAB2)
    solver.stop_sim_time = np.inf
    solver.stop_wall_time = np.inf
    solver.stop_iteration = 50000

    # vorticity & velocity are no longer states of the system. They are true diagnostic variables.
    # But you still might want to set initial condisitons based on vorticity (for example).
    # To do this you'll have to solve for the streamfunction.

    # This will solve for an inital psi, given a vorticity field.
    init = de.LBVP(domain, variables=["init_psi"])

    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    zeta0 = domain.new_field()
    zeta0.set_scales(1)
    x, y = domain.grids(scales=1)

    zeta0["g"] = noise / 40
    zeta0["g"][:, 123:133] += -10
    zeta0["g"][:, 10:15] += 10
    zeta0["g"][:, -15:-10] += 10

    init.parameters["zeta0"] = zeta0
    # init.substitutions["zeta0"] = "  -dy(u0) "
    init.add_equation(
        " d(init_psi,x=2) + d(init_psi,y=2) = zeta0",
        condition="(nx != 0) or  (ny != 0)",
    )
    init.add_equation(" init_psi = 0", condition="(nx == 0) and (ny == 0)")

    init_solver = init.build_solver()
    init_solver.solve()

    psi = solver.state["psi"]
    psi["g"] = init_solver.state["init_psi"]["g"]
    # Now you are ready to go.
    # Anytime you ask for zeta, u, or v they will be non-zero because psy is non-zero.

    dt = 1e-3  # Lx/nx
    # You can set parameters to limit the size of the timestep.
    CFL = flow_tools.CFL(
        solver,
        initial_dt=dt,
        cadence=10,
        safety=2,
        max_change=1.5,
        min_change=0.5,
        max_dt=10 * dt,
    )
    CFL.add_velocities(("u", "v"))

    analysis = solver.evaluator.add_file_handler("analysis", iter=50, max_writes=1000)
    analysis.add_task("-L(psi)", layout="g", name="zeta")
    analysis.add_task("dy(psi)", layout="g", name="u")
    analysis.add_task("-dx(psi)", layout="g", name="v")

    # I don't really know what this is showing. But it makes somethign that looks abotu right.
    plt.ion()
    fig, axis = plt.subplots(figsize=(15, 5))
    p = axis.imshow(zeta0["g"].T, cmap=plt.cm.YlGnBu)
    plt.colorbar(p)
    plt.savefig(f"u0{solver.iteration}.png")

    logger.info("Starting loop")
    while solver.ok:
        # dt = cfl.compute_dt()   # this is returning inf after the first timestep
        # print(dt)
        solver.step(dt)
        print(solver.iteration)
        # if solver.iteration % 10 == 0:
        # This won't work any more.

        # Update plot of scalar field
        # p.set_data(psi["g"].T)
        # p.set_clim(np.min(psi["g"]), np.max(psi["g"]))
        # plt.savefig(f"psi{solver.iteration}.png")
        # There are several ways to see the output as you go.
        # I recommend creating a file handler and saving the output.
        # If you are worried about speed,
        # then plotting in real time is not going to get you to your goals.

    # logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
    # plt.pause(0.001)

    # Print statistics
    # logger.info('Iterations: %i' %solver.iteration)
    logger.info(
        "Iteration: %i, Time: %e, dt: %e" % (solver.iteration, solver.sim_time, dt)
    )


with h5py.File("analysis/analysis_s1/analysis_s1_p0.h5", mode="r") as file:
    zeta = file["tasks"]["zeta"]
    u = file["tasks"]["u"]
    v = file["tasks"]["v"]
    t = zeta.dims[0]["sim_time"]
    x = zeta.dims[1][0]
    y = zeta.dims[2][0]
    # Plot data
    for tstep in range(len(t)):
        print(tstep)
        plt.figure(figsize=(15, 5), dpi=100)
        ct = plt.contourf(
            x[:],
            y[:],
            zeta[tstep, :, :].T,
            cmap="PuOr",
            levels=np.arange(-16, 20, 4),
        )
        plt.quiver(x[::10], y[::10], u[tstep, ::10, ::10].T, v[tstep, ::10, ::10].T)
        plt.colorbar(ct)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Relative vorticity")
        plt.tight_layout()
        plt.savefig(f"zeta{tstep:03d}.png")
        plt.close()
