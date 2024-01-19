using Pkg
#pkg"add GeophysicalFlows, CairoMakie"

using GeophysicalFlows, Printf, Random, CairoMakie
using Random: seed!
using GeophysicalFlows: peakedisotropicspectrum

#Choosing a device
dev = CPU() #CPU() or GPU()

# Set up the model
# Numerical, domain and physical parameters
n, L  = 128, 2π #Grid size and domain size

# time-stepping parameters
dt, tmax = 0.01, 10.0 #time-step and final time
nsteps = Int(tmax/dt) #number of time-steps
nsup = 10 #number of time-steps between outputs

#Problem setup

prob = TwoDNavierStokes.Problem(dev; nx = n, Lx = L, ny = n, Ly = L, dt = dt, stepper = "FilteredRK4")

#We define some shortcuts 

sol, clock, vars, grid = prob.sol, prob.clock, prob.vars, prob.grid
x, y = grid.x, grid.y
Lx, Ly = grid.Lx, grid.L

#Initial conditions
seed!(1234) #set the random seed
k₀, E₀ = 6, 0.5 #initial wavenumber and energy
println("STEP1")

ζ₀ = peakedisotropicspectrum(grid, k₀, E₀, mask=prob.timestepper.filter)
TwoDNavierStokes.set_ζ!(prob, ζ₀)

fig = Figure()
ax = Axis(fig[1, 1];
          xlabel = "x",
          ylabel = "y",
          title = "initial vorticity",
          aspect = 1,
          limits = ((-L/2, L/2), (-L/2, L/2)))

heatmap!(ax, x, y, Array(vars.ζ');
         colormap = :balance, colorrange = (-40, 40))

CairoMakie.save("test.png", fig)