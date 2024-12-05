using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory
using RungeKutta
using BrachiationRobotODE

using LinearAlgebra
using Plots
using Infiltrator
using ForwardDiff

# dynamics of the adaptive system
function f!(xnew, x, u, w, p)
    model = BrachiationRobotODE.Model(9.81, p[1], 0.25, 0.02, p[2], 6e-2, 0.02, 0.28, 0, 0, 0)
    tsit5 = RungeKutta.Tsit5()
    RungeKutta.f!(xnew, tsit5, (ẋ_, x_, u_) -> BrachiationRobotODE.f!(model, ẋ_, x_, u_), x, u, 1e-2)
    xnew .+= w
    return nothing
end

function h(x, v)
    return x[1:2] + v
end

# Horizon, initial state, and inputs
N = 1500
x0 = [pi / 4, 0, 0, 0]
p_accurate = [0.67, 0.72]

# Noise models
## process
μw = zeros(BrachiationRobotODE.nx)
Σw = diagm([1e-4, 1e-3, 1e-3, 1e-2])
invΣw = inv(Σw)

## measurement
μv = zeros(2)
Σv = 1e-6 * I(2)
invΣv = inv(Σv)

# Controller
setpoint(θ, θ̇) = pi / 2 * (1 + sign(sin(θ) * θ̇))

function controller(x)
    q, q̇ = x[1:2], x[3:4]
    P, D = 10, 1.1
    γ_des = setpoint(q[1], q̇[1])
    return -P * (q[2] - γ_des) - D * q̇[2]
end

# random noise
# noise(μ, Σ) = μ + sqrt.(diag(Σ)) .* (rand(length(μ)) .- 0.5)
noise(μ, _) = zeros(length(μ))

# Reference trajectory
x = [zeros(BrachiationRobotODE.nx) for _ in 1:N+1]
z = [zeros(2) for _ in 1:N+1]
u = [zeros(BrachiationRobotODE.nu) for _ in 1:N]

x[1] .= x0
z[1] .= h(x0, μv)

for i in 1:N
    # @infiltrate
    u[i] .= controller(x[i])
    f!(x[i+1], x[i], u[i], noise(μw, Σw), p_accurate)
    z[i+1] .= h(x[i+1], noise(μv, Σv))
end

plt = plot(layout=(2, 1))
plot!(plt, mapreduce(x_ -> x_[1:2]', vcat, x), subplot=1)
plot!(plt, mapreduce(u_ -> u_, vcat, u), subplot=2)

#=
# MHE functions
function dynamics!(ynew, y, w, k)
    @views begin
        xnew = ynew[1:4]
        pnew = ynew[5:6]

        x = y[1:4]
        p = y[5:6]

        wx = w[1:4]
        wp = w[5:6]
    end

    f!(xnew, x, u[k], wx, p)
    pnew .= p + wp
end

function dynamics_diff!(dFdx, dFdu, x, w, k)
    F = similar(x)

    ForwardDiff.jacobian!(dFdx, (xnew, x_) -> dynamics!(xnew, x_, w, k), F, x)
    ForwardDiff.jacobian!(dFdu, (xnew, w_) -> dynamics!(xnew, x, w_, k), F, w)

    return nothing
end

y0 = vcat(x0, p_accurate .* [0.8, 1.2])

function running_cost(y, w, k)
    invΣwp = k == 1 ? diagm([1e-2, 1e0]) : diagm([1e4, 1e6])

    @views begin
        x = y[1:4]

        wx = w[1:4]
        wp = w[5:6]
    end

    dz = z[k] - h(x, μv)
    dw = μw - wx

    return 0.5 * (dz' * invΣv * dz + dw' * invΣw * dw + wp' * invΣwp * wp)
end

function running_cost_diff!(dLdx, dLdu, ddLdxx, ddLdxu, ddLduu, x, w, k)
    ∇xL!(grad, x0, w0) = ForwardDiff.gradient!(grad, (x_) -> running_cost(x_, w0, k), x0)
    ∇uL!(grad, x0, w0) = ForwardDiff.gradient!(grad, (w_) -> running_cost(x0, w_, k), w0)

    ForwardDiff.jacobian!(ddLdxx, (grad, x_) -> ∇xL!(grad, x_, w), dLdx, x)
    ForwardDiff.jacobian!(ddLdxu, (grad, w_) -> ∇xL!(grad, x, w_), dLdx, w)
    ForwardDiff.jacobian!(ddLduu, (grad, w_) -> ∇uL!(grad, x, w_), dLdu, w)

    return nothing
end

function final_cost(y, k)
    x = y[1:4]
    dz = z[k] - h(x, μv)
    return 0.5 * dz' * invΣv * dz
end

function final_cost_diff!(dΦdx, ddΦdxx, x, k)
    result = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(result, x -> final_cost(x, k), x)

    dΦdx .= result.derivs[1]
    ddΦdxx .= result.derivs[2]

    return nothing
end

# plotting callback
function plotting_callback(workset)
    range = 0:workset.N

    # states
    states = mapreduce(x -> x[1:4]', vcat, nominal_trajectory(workset).x)[:, 1:2]
    state_labels = ["x₁", "x₂", "x₃", "x₄"]
    state_plot = plot(range, states, label=permutedims(state_labels))

    state_errors = mapreduce((x_, x_ref_) -> (x_[1:4] - x_ref_)', vcat, nominal_trajectory(workset).x, x)[:, 1:2]
    state_error_labels = ["Δx₁", "Δx₂", "Δx₃", "Δx₄"]
    state_error_plot = plot(range, state_errors, label=permutedims(state_error_labels))

    # parameters
    params = mapreduce(x -> x[5:6]', vcat, nominal_trajectory(workset).x)
    param_labels = ["p₁", "p₂"]
    param_plot = plot(range, params, label=permutedims(param_labels), seriestype=:steppre)

    param_errors = mapreduce(x_ -> ((x_[5:6] - p_accurate) ./ p_accurate)', vcat, nominal_trajectory(workset).x)
    param_error_labels = ["Δp₁/p₁", "Δp₂/p₂"]
    param_error_plot = plot(range, param_errors, label=permutedims(param_error_labels), seriestype=:steppre)

    # disturbances
    dstrb_labels = ["w₁", "w₂", "w₃", "w₄"]
    dstrbs = mapreduce(w -> w', vcat, nominal_trajectory(workset).u)[:, 1:2]
    dstrb_plot = plot(range, vcat(dstrbs, dstrbs[end, :]'), label=permutedims(dstrb_labels), seriestype=:steppost)

    # combination
    plt = plot(state_plot, state_error_plot, param_plot, param_error_plot, dstrb_plot, layout=(5, 1))
    display(plt)

    return plt
end

# optimal estimation
workset = [IterativeLQR.Workset{Float64}(6, 6, n) for n in 1:N] # here nu is actually nw

for i in 1:N
    IterativeLQR.set_initial_state!(workset[i], y0)

    # copy noise estimates from previous iteration and add mean
    if i == 1
        IterativeLQR.set_initial_inputs!(workset[i], [vcat(μw, zeros(2))])
    else
        IterativeLQR.set_initial_inputs!(workset[i], vcat(nominal_trajectory(workset[i-1]).u, [vcat(μw, zeros(2))]))
    end

    IterativeLQR.iLQR!(
        workset[i], dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
        verbose=true, logging=true, plotting_callback=plotting_callback, maxiter=i == N ? 50 : 5
    )
end
=#
