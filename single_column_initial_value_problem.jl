using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using GLMakie

Nz = 500
Lz = 5000
κz = 1e-6

grid = RectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))

vitd = VerticallyImplicitTimeDiscretization()
closure = ScalarDiffusivity(vitd, κ=κz)

gravitational_acceleration = 9.81
equation_of_state = LinearEquationOfState(thermal_expansion=2e-4)
buoyancy = SeawaterBuoyancy(; equation_of_state, gravitational_acceleration, constant_salinity=true)

model = HydrostaticFreeSurfaceModel(; grid, closure, buoyancy, tracers = :T)

Tᵢ(z) = exp(-z^2 / (2 * 100^2))
set!(model, T=Tᵢ)

T = model.tracers.T
z = znodes(T)
lines(interior(T, 1, 1, :), z)

simulation = Simulation(model, Δt=1e3 * days, stop_time=1e7 * days)
run!(simulation)

lines!(interior(T, 1, 1, :), z)
display(current_figure())
