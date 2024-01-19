# %% [markdown]
# # Kelvin-Helmholtz Instability

# %% [markdown]
# We will represent an incompressible, viscous, diffusive, Boussinesq, stratified fluid to simulate a Kelvin-Helmholtz instability.
# 
# <img src="./Stratified_KH_small.jpeg" width="600" height="300" />
# 
# 
# 
# This exercise was designed for the course Waves and Instabilities in Geophysical Fluid Dynamics of the Master's Degree in Advanced Physics and Applied Mathematics, at University of the Balearic Islands (Spain).
# 
# Author: Daniel Argüeso
# Email: d.argueso@uib.es

# %% [markdown]
# 

# %% [markdown]
# ## Import modules

# %%
## Import modules
import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
import matplotlib.pyplot as plt
import h5py
import time
%matplotlib inline
#%matplotlib notebook

# %% [markdown]
# ## Import and set logging

# %% [markdown]
# Here, we set logging to `INFO` level. Currently, by default, Dedalus sets its logging output to `DEBUG`, which produces more info than we need here.

# %%
import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
    
logger = logging.getLogger(__name__)

# %% [markdown]
# ## Define the problem

# %% [markdown]
# ### Set problem domain

# %% [markdown]
# You need to define the domain of the problem. The first two items indicate the aspect ratio. For exmaple (2,1). The second one indicates the number of grid points in each direction. They should be consistent with the aspect ratio

# %%
Lx, Ly = (2., 1.)
nx, ny = (512, 256)

# %% [markdown]
# ### Create bases and domain

# %% [markdown]
# They basically define the transformation between the grid space and the spectral space. There are various types of basis, but the most popular are:
# - Fourier: to define periodic functions in an intervarl (usually the direction of the flow)
# - Chebyshev: general functions in an interval (they require boundary conditions, usually top and bottom)
# 
# For each basis we specify the direction, the dimensions, the interval and the dealising. Dealising is used to evaluate operators in the Fourier space and for numerical stability. We use the default 3/2.
# 
# Then, the domain, which combines both bases and the dimensions above to define the problem domain

# %%
x_basis = de.Fourier("x", nx, interval=(0, Lx), dealias=3 / 2)
y_basis = de.Chebyshev("y", ny, interval=(-Ly / 2, Ly / 2), dealias=3 / 2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# %% [markdown]
# ### Set parameters

# %%
Reynolds = 5e4
g = 9.81
Schmidt = 1e2
Prandtl = 1e2

# %% [markdown]
# ### Define the problem and the equations

# %% [markdown]
# We define the problem. We have different options, but we will use Initial value problem (IVP) in all of our exercises.
# When we define the problem, we have to specify the domain, the variables, the parameters and the equations.
# 
# In the first command, we take the domain specifications we defined before and we define the variables that the problem will use. Here, for example we have pressure, horizontal and vertical velocities, and density. On top of that we also have vertical derivatives of density, and velocities, which we will need to specify the equations.

# %%
#With diffusivity
problem = de.IVP(domain, variables=["p", "u", "uy", "v", "vy", "rho", "rhoy"])
#Without diffusivity
#problem = de.IVP(domain, variables=["p", "u", "uy", "v", "vy", "rho"])


# %%
problem.parameters['Re'] = Reynolds
problem.parameters['g'] = g
problem.parameters['Sc'] = Schmidt 
problem.parameters['Pr'] = Prandtl


# %% [markdown]
# These are the problem equations:

# %% [markdown]
# **Equations**
# 
# See CR 14.2 (PDF version pg 429 - equations pg 431)
# 
# Next we will define the equations that will be solved on this domain.  The equations are
# 
# $$ \partial_t u + \boldsymbol{u}\boldsymbol{\cdot}\boldsymbol{\nabla} u + \frac{\partial_x p}{\rho_0} =  \frac{1}{{\rm Re}} \nabla^2 u $$
# $$ \partial_t v + \boldsymbol{u}\boldsymbol{\cdot}\boldsymbol{\nabla} v + \frac{\partial_y p}{\rho_0} + \frac{\rho g}{\rho_0} =  \frac{1}{{\rm Re}} \nabla^2 v $$
# $$ \boldsymbol{\nabla}\boldsymbol{\cdot}\boldsymbol{u} = 0 $$
# $$ \partial_t \rho + \boldsymbol{u}\boldsymbol{\cdot}\boldsymbol{\nabla} \rho = 0 $$
# 
# The equations are written such that the left-hand side (LHS) is treated implicitly, and the right-hand side (RHS) is treated explicitly.  The LHS is limited to only linear terms, though linear terms can also be placed on the RHS.  Since $y$ is our special direction in this example, we also restrict the LHS to be at most first order in derivatives with respect to $y$.
# 
# **Note**: Note that, unlike the R-T example, there is no diffusivity in the density equation here. You can try to add diffusivity like we did for the R-T example and see what happens. You can use Prandtl number =1. The equation will thus be:
# 
# $$ \partial_t \rho + \boldsymbol{u}\boldsymbol{\cdot}\boldsymbol{\nabla} \rho = \frac{1}{{\rm PrSc}} \nabla^2 \rho $$
#     
# 
# but you need to make some changes to the number of variables and boundary conditions.

# %%
problem.add_equation("dt(u) + dx(p) - 1/Re*(dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy")
problem.add_equation("dt(v) + dy(p) - 1/Re*(dx(dx(v)) + dy(vy)) + g*rho = -u*dx(v) - v*vy")
problem.add_equation("dx(u) + vy = 0")
problem.add_equation("dt(rho) - 1/(Pr*Sc)*(dx(dx(rho)) + dy(rhoy)) = - u*dx(rho) - v*rhoy")
#problem.add_equation("dt(s) = - u*dx(rho) - v*rhoy")
problem.add_equation("vy - dy(v) = 0") 
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("rhoy - dy(rho) = 0")

# %% [markdown]
# ### Define the boundary conditions

# %% [markdown]
# As a general rule, for every derivate on y (our special dimension, Chebyshev), we need to add one boundary conditions. 
# One of them is the pressure gauge.
# 
# Here you will need 5 different boundary conditions, including hte pressure gauge.
# 
# * Two for horizontal velocity (consistent with the bacgkround flow). Dirichlet type, no-slip.
# * Two for vertical velocity. No-slip, no outgoing or incoming flow at the top and bottom walls.
# * Pressure gauge.
# 
# Note that we have a special boundary condition for the $k_x=0$ mode (singled out by `condition="(dx==0)"`).  This is because the continuity equation implies $\partial_y v=0$ if $k_x=0$; thus, $v=0$ on the top and bottom are redundant boundary conditions.  We replace one of these with a gauge choice for the pressure.

# %%
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("integ(p,'y') = 0", condition="(nx == 0)")
problem.add_bc("left(rho) = 0.5")
problem.add_bc("right(rho) = -0.5")

# %% [markdown]
# ## Define the solver

# %% [markdown]
# ### Timestepping

# %% [markdown]
# We have different numerical schemes we can choose from. In our examples, we will use the RK443, but feel free to try others. You may read the documentation to see the full range of options.

# %%
ts = de.timesteppers.RK443

# %% [markdown]
# ### Building the solver

# %% [markdown]
# Here we simply initialize the solver.

# %%
solver =  problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
u = solver.state["u"]
uy = solver.state['uy']
v = solver.state["v"]
vy = solver.state["vy"]
p = solver.state["p"]
rho = solver.state["rho"]
rhoy = solver.state["rhoy"]

# %% [markdown]
# Set the solver parameters

# %% [markdown]
# Here we define some paramters, to help stop the simulation. In our case, we define the maximum duration, but we can define others.

# %%
solver.stop_sim_time = 10.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# %% [markdown]
# Set initial timestep and CFL conditions

# %% [markdown]
# We set the initial timestep, which will be later modified to ensure stability and optimize the simulation depending on the problem itself. We can also define cfl conditions based on velocities.

# %%

initial_dt = 0.2 * Lx / nx
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8,threshold=0.05)
cfl.add_velocities(("u", "v"))

# %% [markdown]
# ### Initial conditions

# %% [markdown]
# Once the problem and the solver are set, we need to describe the initial conditions. These are critical to the problem and may be the difference between success and failure. A right choice of the initial conditions will produce the instability we are studying or simply make the model crash.

# %% [markdown]
# Set initial conditions with a sinusoidal perturbation in vertical velocity

# %%
a = 0.02
amp = -0.2
sigma = 1.0
flow = -1.0
N = 4
u["g"] = flow * np.tanh(4 * y / a) - 0.2
rho["g"] = amp * np.tanh(3 * y / a)
v["g"] = amp * np.exp(-y**2 / sigma**2) * np.sin(N * np.pi * x / Lx)


# %% [markdown]
# Plot initial density and velocities

# %%
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Plot vertical profile of u
axes[0].plot(np.mean(u['g'], axis=0), y[0])
axes[0].set_ylabel('y')
axes[0].set_xlabel('u')
axes[0].set_title('Vertical Profile of u')

# Plot vertical profile of rho
axes[1].plot(np.mean(rho['g'], axis=0), y[0])
axes[1].set_ylabel('y')
axes[1].set_xlabel('rho')
axes[1].set_title('Vertical Profile of rho')

# Plot vertical profile of v
axes[2].plot(v['g'][int(nx/4),:],y[0])
#axes[2].plot(np.mean(v['g'], axis=0), y[0])
axes[2].set_ylabel('y')
axes[2].set_xlabel('v')
axes[2].set_title('Vertical Profile of v')

plt.tight_layout()
plt.show()


# %% [markdown]
# Plot initial vertical velocity

# %%
# plt.close()
# plt.plot(v['g'][int(nx/4),:],y[0])
# plt.ylabel('y')
# plt.xlabel('vertical velocity')
# plt.savefig("KH_velocity.png")

# %% [markdown]
# ## Solving

# %% [markdown]
# In this step run the solver. At the same time we save some information for analysis we may want to make later on. And draw the plots every certain number of timesteps. Saving the analysis is not necessary, but it may be helpful to modify plots without having to run the entire simulation again.

# %% [markdown]
# Prepare the variables that will be saved for analysis (this is optional)

# %%

analysis = solver.evaluator.add_file_handler("analysis", sim_dt=0.1, max_writes=1000)
analysis.add_task("rho")
analysis.add_task("u")
analysis.add_task("v")
analysis.add_task('0.5*(u**2+v**2)',name='KE',scales=(3/2,3/2))
analysis.add_task('0.5*(dx(v)-uy)**2',name='enstrophy')
solver.evaluator.vars['Lx'] = Lx
analysis.add_task("integ(rho,'x')/Lx", name='rho profile')


# %% [markdown]
# ### Plotting initial state

# %% [markdown]
# Make the plot for the initial state
# (remember to use dealias before plotting, see R-T example)

# %%
x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)
fig, axis = plt.subplots(figsize=(8,5))
rho.set_scales(domain.dealias)

p = axis.pcolormesh(xm, ym, rho["g"].T, cmap="RdBu_r")
axis.set_title("Density t = %f" % solver.sim_time)
axis.set_xlim([0,2.])
axis.set_ylim([-0.5,0.5])
#plt.savefig(f'./Kelvin-Helmholtz_instability_000.png')


logger.info('Starting loop')
start_time = time.time()
nt=0
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)
    print(solver.iteration)
    if solver.iteration % 10 == 0:
        # Update plot of scalar field
        p.set_array(rho["g"].T)
        axis.set_title("Density t = %f" % solver.sim_time)
        fig.canvas.draw()
        plt.savefig(f'./Kelvin-Helmholtz_instability_viscous_diffusivity_{nt:03d}.png')
        nt+=1
        
#Video

# def init():
#     axis.set_xlim([0,2.])
#     axis.set_ylim([-0.5,0.5])
#     return p,

# def update(frame):
    
#     if solver.ok:
#         dt = cfl.compute_dt()
#         solver.step(dt)
#         p.set_array(np.ravel(rho['g'][:,:].T))
#         axis.set_title(f't = {solver.sim_time}')
    

# #mimation = FuncAnimation(fig, update,interval = 300)

# anim = animation.FuncAnimation(fig, update, frames=3000, interval=300)
# anim.save('KH.mp4', fps=20)

# %%
# from IPython.display import HTML
# HTML(anim.to_html5_video())

# %% [markdown]
# ### Move into solving loop

# %% [markdown]
# ### ending program with information

# %%
end_time = time.time()
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)

# %% [markdown]
# ## Analysis
# 
# As an example of doing some analysis, we will load in the horizontally averaged profiles of the scalar field $s$ and plot them.

# %%
# # Read in the data
# f = h5py.File('analysis/analysis_s1/analysis_s1_p0.h5','r')
# y = f['/scales/y/1.0'][:]
# t = f['scales']['sim_time'][:]
# rho_ave = f['tasks']['rho profile'][:]
# f.close()

# rho_ave = rho_ave[:,0,:] # remove length-one x dimension

# %%
# fig, axis = plt.subplots(figsize=(8,6))
# for i in range(0,31,6):
#   axis.plot(y,rho_ave[i,:],label='t=%4.2f' %t[i])

# plt.xlim([-0.5,0.5])
# plt.ylim([0,1])
# plt.ylabel(r'$\frac{\int \ rho dx}{L_x}$',fontsize=24)
# plt.xlabel(r'$y$',fontsize=24)
# plt.legend(loc='lower right').draw_frame(False)


