using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory
using CartPoleODE
using MeshCatBenchmarkMechanisms

using ForwardDiff, DiffResults
using Plots
using DataFrames, CSV
using Infiltrator
using BenchmarkTools
using LinearAlgebra
using MatrixEquations

# Cartpole model
cartpole = CartPoleODE.Model(9.81, 1, 0.1, 0.2)

# Horizon and timestep
T = 2
N = 200
h = T / N

# Initial state and inputs
θ₀ = 0 * pi
x₀ = [0, θ₀, 0, 0]
u₀(k) = cos(2 * pi * (k - 1) / N - 1) * ones(CartPoleODE.nu)

# Regularization
regularization = :mchol

# Dynamics
"""RK4 integration with zero-order hold on u"""
function dynamics!(xnew, x, u, _)
    f1 = CartPoleODE.f(cartpole, x, u)
    f2 = CartPoleODE.f(cartpole, x + 0.5 * h * f1, u)
    f3 = CartPoleODE.f(cartpole, x + 0.5 * h * f2, u)
    f4 = CartPoleODE.f(cartpole, x + h * f3, u)
    xnew .= x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return nothing
end

function dynamics_diff!(∇f, x, u, k)
    nx = CartPoleODE.nx
    nu = CartPoleODE.nu

    @views ForwardDiff.jacobian!(
        ∇f,
        (xnew, arg) -> dynamics!(xnew, view(arg, 1:nx), view(arg, nx+1:nx+nu), k),
        zeros(nx),
        vcat(x, u)
    )

    return nothing
end

# Cost functions
ξ(x) = [x[1], cos(x[2] / 2), x[3], x[4]]

Q = h * diagm([1e1, 1e2, 1, 1])
R = h * Matrix{Float64}(I, 1, 1)

S, _ = begin
    x_eq = [0.0, pi, 0, 0]
    u_eq = [0.0]

    nx = CartPoleODE.nx
    nu = CartPoleODE.nu

    ∇f = zeros(nx, nx + nu)
    dynamics_diff!(∇f, x_eq, u_eq, 0)

    E = ForwardDiff.jacobian(ξ, x_eq)
    A = E' * ∇f[:, 1:nx] * inv(E)
    B = E' * ∇f[:, nx+1:nx+nu]

    MatrixEquations.ared(A, B, R, Q)
end

## Running cost
running_cost(x, u, _) = ξ(x)' * Q * ξ(x) + u' * R * u

function running_cost_diff!(∇l, ∇2l, x, u, k)
    nx = CartPoleODE.nx
    nu = CartPoleODE.nu

    H = DiffResults.DiffResult(0.0, (∇l, ∇2l))

    @views ForwardDiff.hessian!(
        H,
        arg -> running_cost(view(arg, 1:nx), view(arg, nx+1:nx+nu), k),
        vcat(x, u)
    )

    return nothing
end

## Final cost
final_cost(x, _) = ξ(x)' * S * ξ(x)

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
workset = IterativeLQR.Workset{Float64}(CartPoleODE.nx, CartPoleODE.nu, N)
IterativeLQR.set_initial_state!(workset, x₀)

IterativeLQR.set_initial_inputs!(workset, [u₀(k) for k in 1:N])
df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, regularization=regularization,
    verbose=true, logging=true, plotting_callback=plotting_callback
)

## Benchmarking
benchmark_iter = findfirst(J -> (J - df.J[end]) <= 1e-2 * df.J[end], df.J)
benchmark_res = @benchmark begin
    IterativeLQR.set_initial_inputs!(workset, [u₀(k) for k in 1:N])
    IterativeLQR.iLQR!(
        workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
        stacked_derivatives=true, regularization=regularization,
        verbose=false, logging=false,
        maxiter=benchmark_iter
    )
end
display(benchmark_res)

bmk = DataFrame(
    "algorithm" => "ilqr",
    "regularization" => "$regularization",
    "nthreads" => Threads.nthreads(),
    "ms" => mean(benchmark_res.times) * 1e-6
)
CSV.write("cartpole/results/cartpole-comp_times.csv", bmk, append=true)

# Save iterations log to csv
CSV.write("cartpole/results/cartpole-ilqr-$regularization.csv", df)

# Save final trajectory
traj = DataFrame(:c => nominal_trajectory(workset).l)

for i in 1:CartPoleODE.nx
    traj[:, Symbol("x$i")] = map(x -> x[i], nominal_trajectory(workset).x)
end

for i in 1:CartPoleODE.nu
    traj[:, Symbol("u$i")] = vcat(map(u -> u[i], nominal_trajectory(workset).u), nominal_trajectory(workset).u[end])
end

CSV.write("cartpole/results/cartpole-trajectory.csv", traj)

# Visualization
(@isdefined vis) || (vis = Visualizer())
render(vis)

## cart-pole
MeshCatBenchmarkMechanisms.set_cartpole!(vis, 0.1, 0.05, 0.05, cartpole.l, 0.02)

## initial configuration
MeshCatBenchmarkMechanisms.set_cartpole_state!(vis, nominal_trajectory(workset).x[1])

## animation
anim = MeshCatBenchmarkMechanisms.Animation(vis, fps=1 / h)
for (i, x) in enumerate(nominal_trajectory(workset).x)
    atframe(anim, i) do
        MeshCatBenchmarkMechanisms.set_cartpole_state!(vis, x)
    end
end
setanimation!(vis, anim, play=false)
