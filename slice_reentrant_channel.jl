using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.Grids: φnodes
using GLMakie

Nx = 1
Ny = 120
Nz = 42

Δz₀ = 10.0 # m
Δz = [Δz₀]
Lz = Δz₀
ϵ = 1.026 # yields Nz = 42
Δzk = Δz₀
z = [-Δz₀]

while Lz < 5000
    global Δzk, Lz
    Δzk = Δzk^ϵ
    push!(Δz, Δzk)
    push!(z, z[end] - Δzk)
    Lz = Lz + Δzk
end

z = reverse(z)
push!(z, 0)
Δz = reverse(Δz)
Nz = length(Δz)

κz = 1e-6
T₀ = 30 # degrees
τ★ = 10days # surface restoring timescale
Tˢ = 0 # degrees
ΔT = 30 # degrees
φ₀ = 60
Δφ = 120
filename = "slice_reentrant_channel.jld2"

# Set up and run the simulation

grid = LatitudeLongitudeGrid(size = (Nx, Ny, Nz),
                             longitude = (-10, 10),
                             latitude = (-60, 60),
                             z = z,
                             topology = (Periodic, Bounded, Bounded))

vitd = VerticallyImplicitTimeDiscretization()
closure = ScalarDiffusivity(vitd, κ=κz)

gravitational_acceleration = 9.81
equation_of_state = LinearEquationOfState(thermal_expansion=2e-4)
buoyancy = SeawaterBuoyancy(; equation_of_state, gravitational_acceleration, constant_salinity=true)

@inline function top_T_flux_func(λ, φ, t, T, p)
    T★ = p.Tˢ + p.ΔT * sin(π * (φ + p.φ₀) / p.Δφ)
    return (T - T★) / p.τ★ 
end

parameters = (; τ★, Tˢ, ΔT, φ₀, Δφ)
top_T_flux = FluxBoundaryCondition(top_T_flux_func; field_dependencies=:T, parameters)
T_bcs = FieldBoundaryConditions(top=top_T_flux)

model = HydrostaticFreeSurfaceModel(; grid, closure, buoyancy,
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    tracers = :T,
                                    boundary_conditions = (; T=T_bcs))

set!(model, T=T₀)

simulation = Simulation(model, Δt=10minutes, stop_time=30days)

outputs = merge(model.velocities, model.tracers)
output_writer = JLD2OutputWriter(model, outputs; filename,
                                 schedule = TimeInterval(1days),
                                 overwrite_existing = true)

simulation.output_writers[:jld2] = output_writer

function progress(sim)
    msg = string("Iter: ", iteration(sim), ", time: ", prettytime(sim))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

run!(simulation)

# Analyze results

Tt = FieldTimeSeries(filename, "T")
ut = FieldTimeSeries(filename, "u")
t = Tt.times
Nt = length(Tt)
λ, φ, z = nodes(Tt)

fig = Figure()
axT = Axis(fig[1, 1], xlabel="Latitude (ᵒ)", ylabel="z (m)")
axu = Axis(fig[1, 2], xlabel="Latitude (ᵒ)", ylabel="z (m)")
slider = Slider(fig[2, 1:2], startvalue=1, range=1:Nt)
n = slider.value

Tn = @lift interior(Tt[$n], 1, :, :)
heatmap!(axT, φ, z, Tn)

un = @lift interior(ut[$n], 1, :, :)
heatmap!(axu, φ, z, un)

title = @lift string("Temperature after ", prettytime(t[$n]))
Label(fig[0, 1:2], title, tellwidth=false)

ylims!(axT, -200, 0)
ylims!(axu, -200, 0)

display(fig)

