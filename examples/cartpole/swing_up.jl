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

# Horizon length, initial state and inputs
T = 2
x₀ = [0, pi * 1e-3, 0, 0]
N = 200

us₀ = [zeros(1) for _ in 1:N]

# Horizon and timestep
h = T / N

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
running_cost(_, u, _) = h * u[1]^2
# running_cost(x, u, _) = 1 + cos(x[2]) + 1e1 * x[1]^2 + 3e-2 * u[1]^2

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
# final_cost(x, _) = 1e5 * (x[1]^2 + (x[2] - pi)^2) + 1e0 * (x[3]^2 + x[4]^2)
final_cost(_, _) = 0

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
    # cost_plot = plot(range, hcat(cumsum(nominal_trajectory(workset).l), nominal_trajectory(workset).p), label=["c" "p"], seriestype=:steppost)

    plt = plot(position_plot, input_plot, cost_plot, layout=(3, 1))
    display(plt)

    return plt
end

# constraint projection functions
terminal_state_projection(_) = [0, pi, 0, 0]
input_projection(u) = map(u_k -> sign(u_k) * min(4.0, abs(u_k)), u)

# Trajectory optimization
workset = IterativeLQR.Workset{Float64}(4, 1, N)
IterativeLQR.set_initial_state!(workset, x₀)
IterativeLQR.set_initial_inputs!(workset, us₀)

IterativeLQR.set_terminal_state_projection_function!(workset, terminal_state_projection)
IterativeLQR.set_input_projection_function!(workset, input_projection)
IterativeLQR.set_terminal_state_constraint_parameter!(workset, 1e0)
IterativeLQR.set_input_constraint_parameter!(workset, 1e-1)

df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, regularization=:min, maxiter=300,
    verbose=true, logging=true, plotting_callback=plotting_callback
)

xN = nominal_trajectory(workset).x[end]
display(xN - terminal_state_projection(xN))

# Benchmark
# opt_J = filter(row -> row.accepted, df).J[end]
# opt_P = filter(row -> row.accepted, df).P[end]
# iter = df.i[findfirst(JP -> (JP - opt_J - opt_P) < 1e-3 * (opt_J + opt_P), df.J .+ df.P)]

# display(@benchmark begin
#     IterativeLQR.set_initial_inputs!(workset, us₀)
#     IterativeLQR.iLQR!(
#         workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
#         stacked_derivatives=true, regularization=:min,
#         verbose=false, maxiter=iter
#     )
# end)

# Visualization
(@isdefined vis) || (vis = Visualizer())
render(vis)

## cart-pole
MeshCatBenchmarkMechanisms.set_cartpole!(vis, 0.1, 0.05, 0.05, model.l, 0.02)

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
