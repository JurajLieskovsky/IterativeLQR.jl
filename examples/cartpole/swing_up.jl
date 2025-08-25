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

# Cartpole model
cartpole = CartPoleODE.Model(9.81, 1, 0.1, 0.2)

# Horizon and timestep
T = 2
N = 200
h = T / N

# Initial state and inputs
θ₀ = 0 * pi
x₀ = [0, θ₀, 0, 0]
u₀(k) = cos((k - 1) / N - 1) * ones(CartPoleODE.nu)

# Algorithm and regularization
algorithm = :ilqr
regularization = :cost

# Dynamics
function dynamics!(xnew, x, u, _)
    CartPoleODE.f!(cartpole, xnew, x, u)
    xnew .*= h
    xnew .+= x

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

function dynamics_diff!(∇f, ∇2f, x, u, k)
    nx = CartPoleODE.nx
    nu = CartPoleODE.nu

    function stacked_dynamics(arg)
        xnew = zeros(eltype(arg), nx)
        dynamics!(xnew, view(arg, 1:nx), view(arg, nx+1:nx+nu), k)
        return xnew
    end

    ForwardDiff.jacobian!(∇2f, (jac, arg) -> ForwardDiff.jacobian!(jac, stacked_dynamics, arg), ∇f, vcat(x, u))

    return nothing
end

# Running cost
Q = diagm([1e1, 1e2, 1, 1])
R = Matrix{Float64}(I, 1, 1)

function running_cost(x, u, _)
    dx = [x[1], cos(x[2] / 2), x[3], x[4]]
    du = u
    return 0.5 * (dx' * Q * dx + du' * R * du)
end

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

# Final cost
function final_cost(x, _)
    S = [
        1220.03 -2933.78 688.133 -206.76
        -2933.78 30350.4 -3136.43 1975.89
        688.133 -3136.43 683.13 -220.279
        -206.76 1975.89 -220.279 138.723
    ]
    dx = [x[1], cos(x[2] / 2), x[3], x[4]]
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
    stacked_derivatives=true, regularization=regularization, algorithm=algorithm,
    verbose=true, logging=true, plotting_callback=plotting_callback
)

CSV.write("cartpole/results/cartpole-$algorithm.csv", df)

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
