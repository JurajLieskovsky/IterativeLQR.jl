import IterativeLQR
import IterativeLQR: nominal_trajectory

include("CartPole.jl")
using .CartPole
include("RungeKutta.jl")
using .RungeKutta

using ForwardDiff
using Plots

# Horizon and timestep
T = 2
N = 100
h = T/N

# Dynamics
model = CartPole.Model(9.8, 1, 0.5, 0.1)
f!(dx, x, u) = CartPole.f!(model, dx, x, u)

rk4 = RungeKutta.RK4()
dynamics!(dx, x, u) = RungeKutta.f!(dx, rk4, f!, x, u, h)

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
final_cost(x) = 1 + cos(x[2]) + 1e-1 * x[1]^2 + 1e0*x[3]^2 + 1e0*x[4]^2

function final_cost_diff!(dΦdx, ddΦdxx, x)
    result = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(result, final_cost, x)

    dΦdx .= result.derivs[1]
    ddΦdxx .= result.derivs[2]

    return nothing
end

# iLQR workset and initial guess
workset = IterativeLQR.Workset{Float64}(4, 1, N)

IterativeLQR.set_initial_state!(workset, [0.0, 0, 0, 0])
IterativeLQR.set_initial_inputs!(workset, [[1e-3] for _ in 1:N])

# Trajectory optimization
IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    verbose=true
)

# Plotting
position_plot = plot()
for i = 1:2
    state_series = [x[i] for x in nominal_trajectory(workset).x]
    plot!(position_plot, 0:N, state_series, label="x"*string(i))
end

input_plot = plot()
input_series = [u[1] for u in nominal_trajectory(workset).u]
plot!(input_plot, 0:N, vcat(input_series, input_series[end]), label="u", seriestype=:steppost)

cost_plot = plot()
plot!(cost_plot, 0:N, cumsum(nominal_trajectory(workset).l), label="c", seriestype=:steppost)

plot(position_plot, input_plot, cost_plot, layout=(3,1))
