using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory
using RungeKutta
using CartPoleODE

using ForwardDiff
using Plots
using DataFrames, CSV

# Horizon and timestep
T = 2
N = 150
h = T / N

# Initial state and inputs
x₀ = zeros(4)
us₀ = [[1e-3] for _ in 1:N]
# us₀ = [[1e-4 * 2 * pi * k / N] for k in 0:N-1]

# Dynamics
model = CartPoleODE.Model(9.81, 1, 0.1, 0.2)
f!(dx, x, u) = dx .= CartPoleODE.f(model, x, u)

tsit5 = RungeKutta.Tsit5()
dynamics!(dx, x, u, _) = RungeKutta.f!(dx, tsit5, f!, x, u, h)

function dynamics_diff!(fx, fu, x, u, k)
    nx = CartPoleODE.nx
    nu = CartPoleODE.nu

    arg = vcat(x, u)
    res = zeros(nx)
    jac = zeros(nx, nx+nu)

    @views begin
        ForwardDiff.jacobian!(jac, (xnew, arg) -> dynamics!(xnew, arg[1:nx], arg[nx+1:nx+nu], k), res, arg)

        fx .= jac[:, 1:nx]
        fu .= jac[:, nx+1:nx+nu]
    end

    return nothing
end

# Running cost
running_cost(_, u, _) = 1e-2 * h * u[1]^2

function running_cost_diff!(dLdx, dLdu, ddLdxx, ddLdxu, ddLduu, x, u, k)
    ∇xL!(grad, x0, u0) = ForwardDiff.gradient!(grad, (x_) -> running_cost(x_, u0, k), x0)
    ∇uL!(grad, x0, u0) = ForwardDiff.gradient!(grad, (u_) -> running_cost(x0, u_, k), u0)

    ForwardDiff.jacobian!(ddLdxx, (grad, x_) -> ∇xL!(grad, x_, u), dLdx, x)
    ForwardDiff.jacobian!(ddLdxu, (grad, u_) -> ∇xL!(grad, x, u_), dLdx, u)
    ForwardDiff.jacobian!(ddLduu, (grad, u_) -> ∇uL!(grad, x, u_), dLdu, u)

    return nothing
end

# Final cost
final_cost(x, _) = 1e2 * (1 + cos(x[2])) + 1e1 * x[1]^2 + 1e2 * x[3]^2 + 1e2 * x[4]^2

function final_cost_diff!(dΦdx, ddΦdxx, x, k)
    result = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(result, x -> final_cost(x, k), x)

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
    input_plot = plot(range, vcat(inputs, inputs[end, :]'), label="u", seriestype=:steppost)

    cost_plot = plot(range, cumsum(nominal_trajectory(workset).l), label="c", seriestype=:steppost)

    plt = plot(position_plot, input_plot, cost_plot, layout=(3, 1))
    display(plt)

    return plt
end

# Trajectory optimization
df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    verbose=true, logging=true, plotting_callback=plotting_callback
)

df[!, :bwd] .= N
df[!, :fwd] .= N
CSV.write("cartpole/results/ilqr-iterations.csv", df)
