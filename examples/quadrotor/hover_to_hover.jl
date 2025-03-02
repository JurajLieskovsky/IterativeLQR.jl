using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory, active_trajectory
using RungeKutta
using QuadrotorODE
using MeshCatBenchmarkMechanisms

using LinearAlgebra
using ForwardDiff
using Plots
using DataFrames, CSV
using BenchmarkTools

# Horizon and timestep
T = 4
N = 400
h = T / N

# Target state
xₜ = vcat([0, 0, 2], [1, 0, 0, 0], zeros(3), zeros(3))

# Initial state and inputs
x₀ = vcat([6, -6, 1], [1, 0, 0, 0], zeros(3), zeros(3))
u₀ = 9.81 / 4 * ones(4)
us₀ = [u₀ for _ in 1:N]

# Dynamics
a = 0.3
quadrotor = QuadrotorODE.System([0, 0, -9.81], 1, I(3), a, 0.01)

tsit5 = RungeKutta.Tsit5()

dynamics!(xnew, x, u, _) = RungeKutta.f!(
    xnew,
    tsit5,
    (xnew_, x_, u_) -> xnew_ .= QuadrotorODE.dynamics(quadrotor, x_, u_),
    x,
    u,
    h
)

function dynamics_diff!(jac, x, u, _)
    f!(dznew, x, dz, u) = RungeKutta.f!(
        dznew,
        tsit5,
        (dznew_, dz_, u_) -> dznew_ .= QuadrotorODE.tangential_dynamics(quadrotor, x, dz_, u_),
        dz,
        u,
        h
    )

    nz = QuadrotorODE.nz
    nu = QuadrotorODE.nu

    @views ForwardDiff.jacobian!(
        jac,
        (dznew, arg) -> f!(dznew, x, arg[1:nz], arg[nz+1:nz+nu]),
        zeros(nz),
        vcat(zeros(nz), u)
    )
end

# Running cost
running_cost(_, u, _) = h * (2e-2 * u' * u - 1e-1 * sum(log.(u)))

function running_cost_diff!(grad, hess, x, u, k)
    nz = QuadrotorODE.nz
    nu = QuadrotorODE.nu

    H = DiffResults.DiffResult(0.0, (grad, hess))

    @views ForwardDiff.hessian!(
        H,
        arg -> running_cost(QuadrotorODE.incremented_state(x, arg[1:nz]), arg[nz+1:nz+nu], k),
        vcat(zeros(nz), u)
    )

    return nothing
end

# Final cost
function final_cost(x, _)
    dx = QuadrotorODE.state_difference(x, xₜ)
    return dx' * diagm(vcat(1e3 * ones(6), 1e2 * ones(6))) * dx
end

function final_cost_diff!(Φx, Φxx, x, k)
    dz = zeros(12)
    result = DiffResults.HessianResult(dz)
    ForwardDiff.hessian!(result, dz_ -> final_cost(QuadrotorODE.incremented_state(x, dz_), k), dz)

    Φx .= result.derivs[1]
    Φxx .= result.derivs[2]

    return nothing
end

# Plotting callback
function plotting_callback(workset)
    range = 0:workset.N

    states = mapreduce(x -> x', vcat, nominal_trajectory(workset).x)
    state_labels = ["x" "y" "z" "q₀" "q₁" "q₂" "q₃" "vx" "vy" "vz" "ωx" "ωy" "ωz"]
    position_plot = plot(range, states[:, 1:7], label=state_labels[1:1, 1:7])

    inputs = mapreduce(u -> u', vcat, nominal_trajectory(workset).u)
    input_labels = ["u₀" "u₁" "u₂" "u₃"]
    input_plot = plot(range, vcat(inputs, inputs[end, :]'), label=input_labels, seriestype=:steppost)

    cost_plot = plot(range, cumsum(nominal_trajectory(workset).l), label="c", seriestype=:steppost)

    plt = plot(position_plot, input_plot, cost_plot, layout=(3, 1))
    display(plt)

    return plt
end

# Trajectory optimization
workset = IterativeLQR.Workset{Float64}(13, 4, N, 12)
IterativeLQR.set_initial_state!(workset, x₀)
IterativeLQR.set_initial_inputs!(workset, us₀)

df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!, stacked_derivatives=true,
    verbose=true, logging=true, plotting_callback=plotting_callback, state_difference=QuadrotorODE.state_difference
)

# Benchmark
opt = filter(row -> row.accepted, df).J[end]
iter = df.i[findfirst(J -> (J - opt) < 1e-3 * opt, df.J)] 

bench = @benchmark begin 
    IterativeLQR.set_initial_inputs!(workset, us₀)
    IterativeLQR.iLQR!(
        workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!, stacked_derivatives=true,
        verbose=false, maxiter=iter,
        state_difference=QuadrotorODE.state_difference
    )
end

display(bench)

# Visualization
vis = (@isdefined vis) ? vis : Visualizer()
render(vis)

## quadrotor and target
MeshCatBenchmarkMechanisms.set_quadrotor!(vis, 2 * a, 0.12, 0.25)
MeshCatBenchmarkMechanisms.set_target!(vis, 0.12)

## initial configuration
MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, nominal_trajectory(workset).x[1])
MeshCatBenchmarkMechanisms.set_target_position!(vis, xₜ[1:3])

## animation
fps = 100
anim = MeshCatBenchmarkMechanisms.Animation(vis, fps=fps)
for (i, x) in enumerate(nominal_trajectory(workset).x)
    atframe(anim, i) do
        MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, x)
    end
end
setanimation!(vis, anim, play=false);

df[!, :bwd] .= N
df[!, :fwd] .= N
CSV.write("quadrotor/results/ilqr-iterations.csv", df)
