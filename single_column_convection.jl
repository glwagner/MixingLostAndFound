using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using GLMakie

Nz = 500
Lz = 5000
κz = 1e-6
T₀ = 30 # degrees
τ★ = 10days # surface restoring timescale
T★ = 5 # degrees
filename = "single_column_convection.jld2"

# Set up and run the simulation

grid = RectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))

vitd = VerticallyImplicitTimeDiscretization()
closure = ScalarDiffusivity(vitd, κ=κz)

gravitational_acceleration = 9.81
equation_of_state = LinearEquationOfState(thermal_expansion=2e-4)
buoyancy = SeawaterBuoyancy(; equation_of_state, gravitational_acceleration, constant_salinity=true)

@inline top_T_flux_func(t, T, p) = (T - p.T★) / p.τ★ 
top_T_flux = FluxBoundaryCondition(top_T_flux_func, field_dependencies=:T, parameters=(; τ★, T★))
T_bcs = FieldBoundaryConditions(top=top_T_flux)

model = HydrostaticFreeSurfaceModel(; grid, closure, buoyancy,
                                    tracers = :T,
                                    boundary_conditions = (; T=T_bcs))

set!(model, T=T₀)

simulation = Simulation(model, Δt=1day, stop_time=1e3 * days)

output_writer = JLD2OutputWriter(model, model.tracers; filename,
                                 schedule = TimeInterval(10days),
                                 overwrite_existing = true)

simulation.output_writers[:jld2] = output_writer

run!(simulation)

# Analyze results

Tt = FieldTimeSeries(filename, "T")
t = Tt.times
Nt = length(Tt)

fig = Figure()
axT = Axis(fig[1, 1], xlabel="Temperature (ᵒC)", ylabel="z (m)")
slider = Slider(fig[2, 1], startvalue=1, range=1:Nt)
n = slider.value

Tn = @lift interior(Tt[$n], 1, 1, :)
lines!(axT, Tn, z)

title = @lift string("Temperature after ", prettytime(t[$n]))
Label(fig[0, 1], title, tellwidth=false)

ylims!(axT, -100, 0)

display(fig)

