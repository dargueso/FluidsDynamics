using FourierFlows, CairoMakie, Printf, Random, JLD2
using LinearAlgebra: mul!, ldiv!


struct Params{T} <: AbstractParams
    g :: T # Gravitational acceleration
end

nothing # hide

struct Vars{Aphys, Atrans} <: AbstractVars
    u :: Aphys # Velocity
    v :: Aphys # Velocity
    p :: Aphys # Pressure
    ρ :: Aphys # Density

    uh :: Atrans # Velocity
    vh :: Atrans # Velocity
    ph :: Atrans # Pressure
    ρh :: Atrans # Density
    
end

nothing # hide

# A constructor populates empty arrays based on the dimension of the `grid`
# and then creates `Vars` struct.
"""
    Vars(grid)

Create a `Vars` struct with arrays of the appropriate size for the given `grid`.
"""

function Vars(grid)
    Dev = typeof(grid)
    T = typeof(grid)

    @devzeros Dev T grid.nx u v p ρ
    @devzeros Dev Complex{T} grid.nkr uh vh ph ρh

    return Vars(u, v, p, ρ, uh, vh, ph, ρh)
end

nothing # hide

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Compute the nonlinear terms.
"""

function calcN!(N, sol, t, clock, vars, params, grid)



"""
    Equation(params, grid)

Construct the equation: the linear part and the nonlinear part, which is computed by `calcN!`
"""

function Equation(params, grid)
    T = eltype(grid)
    dev = grid.device

    L = zeros(dev, T, (grid.nkr, 4))
    D = @. - 

    @devzeros Dev T grid.nx u v p ρ
    @devzeros Dev Complex{T} grid.nkr uh vh ph ρh

    return Vars(u, v, p, ρ, uh, vh, ph, ρh)
end


# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)
nothing # hide

# ## Numerical parameters and time-stepping parameters

nx = 512            # grid resolution
stepper = "FilteredRK4"  # timestepper
     dt = 20.0           # timestep (s)
 nsteps = 320            # total number of time-steps
nothing # hide

# ## Domain and physical parameters

g  = 9.8        # Gravitational acceleration (m s⁻²)

grid = Grid(nx, ny, Lx, Ly, dev) # Grid
params = Params(g) # Physical parameters
vars = Vars(grid) # Variables
equation = Equation(params, grid) # Equation

prob = FourierFlows.Problem(equation, stepper, dt, grid, vars, params) # Problem

nothing # hide

# ## Initial conditions

# Random seed

seed!(1234)


