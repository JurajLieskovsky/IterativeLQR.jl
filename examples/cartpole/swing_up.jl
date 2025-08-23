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

# Cartpole model
cartpole = CartPoleODE.Model(9.81, 1, 0.1, 0.2)

# Horizon and timestep
T = 2
N = 200
h = T / N

# Initial state and inputs
u₀ = zeros(CartPoleODE.nu)
x₀ = [0, pi * 1e-3, 0, 0]

# Algorithm and regularization
algorithm = :ddp
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
# running_cost(_, u, _) = 1e-2 * h * u[1]^2
running_cost(x, u, _) = 1 + cos(x[2]) + 1e1 * x[1]^2 + 5e-2 * u[1]^2

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
# final_cost(x, _) = 1e3 * (x[1]^2 + (x[2] - pi)^2) + 1e0 * (x[3]^2 + x[4]^2)
final_cost(x, _) = 1 + cos(x[2]) + 1e1 * x[1]^2

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

IterativeLQR.set_initial_inputs!(workset, [u₀ for _ in 1:N])
df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, regularization=regularization, algorithm=algorithm,
    verbose=true, logging=true, plotting_callback=plotting_callback
)

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
