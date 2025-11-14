using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory
using UCNCartPoleODE
using MeshCatBenchmarkMechanisms

using ForwardDiff, DiffResults
using Plots
using DataFrames, CSV
using Infiltrator
using BenchmarkTools
using LinearAlgebra
using MatrixEquations

# Cartpole model
cartpole = UCNCartPoleODE.Model(9.81, 1, 0.1, 0.2)

# Horizon and timestep
T = 2
N = 200
h = T / N

# Target state
xₜ = [0.0, 0, 1, 0, 0]
uₜ = zeros(UCNCartPoleODE.nu)

# Initial state and inputs
θ₀ = 0 * pi
x₀ = [0, cos(θ₀ / 2), sin(θ₀ / 2), 0, 0]
u₀(k) = cos(2 * pi * (k - 1) / N - 1) * ones(UCNCartPoleODE.nu)

# Dynamics
"""RK4 integration with zero-order hold on u"""
function dynamics!(xnew, x, u, _)
    f1 = UCNCartPoleODE.f(cartpole, x, u)
    f2 = UCNCartPoleODE.f(cartpole, x + 0.5 * h * f1, u)
    f3 = UCNCartPoleODE.f(cartpole, x + 0.5 * h * f2, u)
    f4 = UCNCartPoleODE.f(cartpole, x + h * f3, u)
    xnew .= x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    UCNCartPoleODE.normalize_state!(xnew)
    return nothing
end

function dynamics_diff!(∇f, x, u, k)
    nx = UCNCartPoleODE.nx
    nu = UCNCartPoleODE.nu

    @views ForwardDiff.jacobian!(
        ∇f,
        (xnew, arg) -> dynamics!(xnew, view(arg, 1:nx), view(arg, nx+1:nx+nu), k),
        zeros(nx),
        vcat(x, u)
    )

    return nothing
end

# Running cost
Q = h * diagm([1e1, 1e2, 1, 1])
R = h * Matrix{Float64}(I, 1, 1)

function running_cost(x, u, _)
    dx = UCNCartPoleODE.state_difference(x, xₜ)
    du = u - uₜ
    return dx' * Q * dx + du' * R * du
end

function running_cost_diff!(∇l, ∇2l, x, u, k)
    nx = UCNCartPoleODE.nx
    nu = UCNCartPoleODE.nu

    H = DiffResults.DiffResult(0.0, (∇l, ∇2l))

    @views ForwardDiff.hessian!(
        H,
        arg -> running_cost(view(arg, 1:nx), view(arg, nx+1:nx+nu), k),
        vcat(x, u)
    )

    return nothing
end

# Final cost
S, _ = begin
    nx = UCNCartPoleODE.nx
    nu = UCNCartPoleODE.nu

    ∇f = zeros(nx, nx + nu)
    dynamics_diff!(∇f, xₜ, uₜ, 0)

    E = UCNCartPoleODE.jacobian(xₜ)
    A = E' * ∇f[:, 1:nx] * E
    B = E' * ∇f[:, nx+1:nx+nu]

    MatrixEquations.ared(A, B, R, Q)
end

function final_cost(x, _)
    dx = UCNCartPoleODE.state_difference(x, xₜ)
    return dx' * S * dx
end

function final_cost_diff!(Φx, Φxx, x, k)
    H = DiffResults.DiffResult(0.0, (Φx, Φxx))
    ForwardDiff.hessian!(H, x -> final_cost(x, k), x)
    return nothing
end

# Plotting callback
function plotting_callback(workset)
    range = 0:workset.N

    states = mapreduce(x -> x', vcat, nominal_trajectory(workset).x)
    state_labels = ["x₁" "x₂" "x₃" "x₄" "x₅"]
    position_plot = plot(range, states[:, 1:3], label=state_labels[1:1, 1:3])

    inputs = mapreduce(u -> u', vcat, nominal_trajectory(workset).u)
    input_plot = plot(range, vcat(inputs, inputs[end, :]'), label="u", seriestype=:steppost)

    cost_plot = plot(range, cumsum(nominal_trajectory(workset).l), label="c", seriestype=:steppost)

    plt = plot(position_plot, input_plot, cost_plot, layout=(3, 1))
    display(plt)

    return plt
end

# Trajectory optimization
workset = IterativeLQR.Workset{Float64}(UCNCartPoleODE.nx, UCNCartPoleODE.nu, N, UCNCartPoleODE.nd)
IterativeLQR.set_initial_state!(workset, x₀)

IterativeLQR.set_nominal_inputs!(workset, [u₀(k) for k in 1:N])
df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, regularization=:none,
    coordinate_jacobian=UCNCartPoleODE.jacobian, state_difference=UCNCartPoleODE.state_difference,
    verbose=true, logging=true, plotting_callback=plotting_callback,
)

# Save iterations log to csv
# CSV.write("ucn_cartpole/results/iterations.csv", df)

# Visualization
(@isdefined vis) || (vis = Visualizer())
render(vis)

## utility for converting to a state that uses an angle instead of a UCN
xq2xθ(x) = [x[1], 2 * atan(x[3], x[2]), x[4], x[5]]

## cart-pole
MeshCatBenchmarkMechanisms.set_cartpole!(vis, 0.1, 0.05, 0.05, cartpole.l, 0.02)

## initial configuration
MeshCatBenchmarkMechanisms.set_cartpole_state!(vis, xq2xθ(nominal_trajectory(workset).x[1]))

## animation
anim = MeshCatBenchmarkMechanisms.Animation(vis, fps=1 / h)
for (i, x) in enumerate(nominal_trajectory(workset).x)
    atframe(anim, i) do
        MeshCatBenchmarkMechanisms.set_cartpole_state!(vis, xq2xθ(x))
    end
end
setanimation!(vis, anim, play=false)
