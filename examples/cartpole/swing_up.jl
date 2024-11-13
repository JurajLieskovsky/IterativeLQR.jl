using Revise

import IterativeLQR
import IterativeLQR: nominal_trajectory

using RungeKutta
include("CartPole.jl")
using .CartPole

using ForwardDiff
using Plots

# Horizon and timestep
T = 2
N = 200
h = T / N

# Initial state and inputs
x₀ = zeros(4)
us₀ = [[1e-3] for _ in 1:N]
# us₀ = [[1e-4 * 2 * pi * k / N] for k in 0:N-1]

# Dynamics
model = CartPole.Model(9.8, 1, 0.5, 0.1)
f!(dx, x, u) = CartPole.f!(model, dx, x, u)

tsit5 = RungeKutta.Tsit5()
dynamics!(dx, x, u) = RungeKutta.f!(dx, tsit5, f!, x, u, h)

function dynamics_diff!(dFdx, dFdu, x, u)
    F = similar(x)

    ForwardDiff.jacobian!(dFdx, (xnew, x_) -> dynamics!(xnew, x_, u), F, x)
    ForwardDiff.jacobian!(dFdu, (xnew, u_) -> dynamics!(xnew, x, u_), F, u)

    return nothing
end

# Running cost
running_cost(_, u) = 1e-4 * h * u[1]^2

function running_cost_diff!(dLdx, dLdu, ddLdxx, ddLdxu, ddLduu, x, u)
    ∇xL!(grad, x0, u0) = ForwardDiff.gradient!(grad, (x_) -> running_cost(x_, u0), x0)
    ∇uL!(grad, x0, u0) = ForwardDiff.gradient!(grad, (u_) -> running_cost(x0, u_), u0)

    ForwardDiff.jacobian!(ddLdxx, (grad, x_) -> ∇xL!(grad, x_, u), dLdx, x)
    ForwardDiff.jacobian!(ddLdxu, (grad, u_) -> ∇xL!(grad, x, u_), dLdx, u)
    ForwardDiff.jacobian!(ddLduu, (grad, u_) -> ∇uL!(grad, x, u_), dLdu, u)

    return nothing
end

# Final cost
final_cost(x) = 1 + cos(x[2]) + 1e-1 * x[1]^2 + 1e0 * x[3]^2 + 1e0 * x[4]^2

function final_cost_diff!(dΦdx, ddΦdxx, x)
    result = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(result, final_cost, x)

    dΦdx .= result.derivs[1]
    ddΦdxx .= result.derivs[2]

    return nothing
end

# iLQR workset and initial guess
workset = IterativeLQR.Workset{Float64}(4, 1, N)
IterativeLQR.set_initial_state!(workset, x₀)
IterativeLQR.set_initial_inputs!(workset, us₀)

# Plotting callback
function plotting_callback(workset)
    range = 0:workset.N

    states = mapreduce(x -> x', vcat, nominal_trajectory(workset).x)
    state_labels = ["x₁" "x₂" "x₃" "x₄"]
    position_plot = plot(range, states[:, 1:2], label=state_labels[1:1, 1:2])

    inputs = mapreduce(u -> u', vcat, nominal_trajectory(workset).u)
    input_plot = plot(range, vcat(inputs, inputs[end,:]'), label="u", seriestype=:steppost)

    cost_plot = plot(range, cumsum(nominal_trajectory(workset).l), label="c", seriestype=:steppost)

    plt = plot(position_plot, input_plot, cost_plot, layout=(3, 1))
    display(plt)

    return plt
end

# Trajectory optimization
IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    verbose=true, plotting_callback=plotting_callback
)
