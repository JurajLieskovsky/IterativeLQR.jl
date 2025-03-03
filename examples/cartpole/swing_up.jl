using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory
using CartPoleODE

using ForwardDiff, DiffResults
using Plots
using DataFrames, CSV
using Infiltrator
using BenchmarkTools

# Horizon and timestep
T = 2
N = 200
h = T / N

# Initial state and inputs
x₀ = [0, pi * 1e-3, 0, 0]
us₀ = [zeros(1) for _ in 1:N]

# Dynamics
model = CartPoleODE.Model(9.81, 1, 0.1, 0.2)

function dynamics!(xnew, x, u, _)
    CartPoleODE.f!(model, xnew, x, u)
    xnew .*= h
    xnew .+= x

    return nothing
end

function dynamics_diff!(jac, x, u, k)
    nx = CartPoleODE.nx
    nu = CartPoleODE.nu

    @views ForwardDiff.jacobian!(
        jac,
        (xnew, arg) -> dynamics!(xnew, arg[1:nx], arg[nx+1:nx+nu], k),
        zeros(nx),
        vcat(x, u)
    )

    return nothing
end

# Running cost
running_cost(_, u, _) = 1e-2 * h * u[1]^2

function running_cost_diff!(grad, hess, x, u, k)
    nx = CartPoleODE.nx
    nu = CartPoleODE.nu

    H = DiffResults.DiffResult(0.0, (grad, hess))

    @views ForwardDiff.hessian!(
        H,
        arg -> running_cost(arg[1:nx], arg[nx+1:nx+nu], k),
        vcat(x, u)
    )

    return nothing
end

# Final cost
final_cost(x, _) = 1e2 * (x[1]^2 + (x[2] - pi)^2 + x[3]^2 + x[4]^2)

function final_cost_diff!(Φx, Φxx, x, k)
    H = DiffResults.DiffResult(0.0, (Φx, Φxx))
    ForwardDiff.hessian!(H, x -> final_cost(x, k), x)
    return nothing
end

# Plotting callback
function plotting_callback(workset)
    range = 0:workset.N

    states = mapreduce(x -> x', vcat, nominal_trajectory(workset).x)
    state_labels = ["x₁" "x₂" "x₃" "x₄"]
    position_plot = plot(range, states[:, 1:2], label=state_labels[1:1, 1:2])

    inputs = mapreduce(u -> u', vcat, nominal_trajectory(workset).u)
    input_plot = plot(range, vcat(inputs, inputs[end, :]'), label="u", seriestype=:steppost)

    cost_plot = plot(range, cumsum(nominal_trajectory(workset).l), label="c", seriestype=:steppost)

    plt = plot(position_plot, input_plot, cost_plot, layout=(3, 1))
    display(plt)

    return plt
end

# Trajectory optimization
workset = IterativeLQR.Workset{Float64}(4, 1, N)
IterativeLQR.set_initial_state!(workset, x₀)

IterativeLQR.set_initial_inputs!(workset, us₀)
df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, regularization=:holy,
    verbose=true, logging=true, plotting_callback=plotting_callback
)

# Benchmark
opt = filter(row -> row.accepted, df).J[end]
iter = df.i[findfirst(J -> (J - opt) < 1e-3 * opt, df.J)] 

bench = @benchmark begin 
    IterativeLQR.set_initial_inputs!(workset, us₀)
    IterativeLQR.iLQR!(
        workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
        stacked_derivatives=true, regularization=:holy,
        verbose=false, maxiter=iter
    )
end

display(bench)

df[!, :bwd] .= N
df[!, :fwd] .= N
CSV.write("cartpole/results/ilqr-iterations.csv", df)
