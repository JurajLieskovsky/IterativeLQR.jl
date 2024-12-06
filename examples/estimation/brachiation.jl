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
N = 300
x0 = [pi / 4, 0, 0, 0]
p_accurate = [0.67, 0.72]

# Noise models
## process
μw = zeros(BrachiationRobotODE.nx)
Σw = diagm([1e-5, 1e-4, 1e-4, 1e-3])
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
    u[i] .= controller(x[i])
    f!(x[i+1], x[i], u[i], noise(μw, Σw), p_accurate)
    z[i+1] .= h(x[i+1], noise(μv, Σv))
end

# Differentiation functions
function dynamics_diff!(dynamics!, fx, fu, x, w, k)
    f = similar(x)

    ForwardDiff.jacobian!(fx, (xnew, x_) -> dynamics!(xnew, x_, w, k), f, x)
    ForwardDiff.jacobian!(fu, (xnew, w_) -> dynamics!(xnew, x, w_, k), f, w)

    return nothing
end

function running_cost_diff!(running_cost, lx, lu, lxx, lxu, luu, x, w, k)
    ∇xL!(grad, x0, w0) = ForwardDiff.gradient!(grad, (x_) -> running_cost(x_, w0, k), x0)
    ∇uL!(grad, x0, w0) = ForwardDiff.gradient!(grad, (w_) -> running_cost(x0, w_, k), w0)

    ForwardDiff.jacobian!(lxx, (grad, x_) -> ∇xL!(grad, x_, w), lx, x)
    ForwardDiff.jacobian!(lxu, (grad, w_) -> ∇xL!(grad, x, w_), lx, w)
    ForwardDiff.jacobian!(luu, (grad, w_) -> ∇uL!(grad, x, w_), lu, w)

    return nothing
end

function final_cost_diff!(final_cost, Φx, Φxx, x, k)
    result = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(result, x -> final_cost(x, k), x)

    Φx .= result.derivs[1]
    Φxx .= result.derivs[2]

    return nothing
end

# Moving horizon estimation
p0 = p_accurate .* [1.2, 0.8] # false assumption on parameters
initial_state = vcat(x0, p0)
noise_estimate = [ vcat(μw, zeros(2)) for _ in 1:N]

workset = IterativeLQR.Workset{Float64}(6, 6, N)
IterativeLQR.set_initial_state!(workset, initial_state)
IterativeLQR.set_initial_inputs!(workset, noise_estimate)

M = 20 # observation horizon

function dynamics!(x̂new, x̂, û, k, n=0)
    @views begin
        xnew = x̂new[1:4]
        pnew = x̂new[5:6]

        x = x̂[1:4]
        p = x̂[5:6]
        w = û[1:4]
        q = û[5:6]
    end

    f!(xnew, x, u[k+n], w, p)
    pnew .= p + q
end

function running_cost(x̂, û, k, invΣq, n=0)
    @views begin
        x = x̂[1:4]
        w = û[1:4]
        q = û[5:6]
    end

    dz = z[k+n] - h(x, μv)
    dw = μw - w

    return 0.5 * (dz' * invΣv * dz + dw' * invΣw * dw + q' * invΣq * q)
end

function final_cost(x̂, k, n=0)
    x = x̂[1:4]
    dz = z[k+n] - h(x, μv)
    return 0.5 * dz' * invΣv * dz
end

## plotting callback
function plotting_callback(workset, n=0)
    range = 0:workset.N

    states = mapreduce(x -> x', vcat, nominal_trajectory(workset).x)
    state_labels = ["x₁", "x₂", "x₃", "x₄", "p₁", "p₂"]
    state_plot = plot(range, states, label=permutedims(state_labels))

    errors = mapreduce(
        (x_, x_ref_) -> (x_ - vcat(x_ref_, p_accurate))',
        vcat,
        nominal_trajectory(workset).x, circshift(x, -n)
    )
    error_labels = ["Δx₁", "Δx₂", "Δx₃", "Δx₄", "Δp₁", "Δp₂"]
    error_plot = plot(range, errors, label=permutedims(error_labels))

    dstrb_labels = ["w₁", "w₂", "w₃", "w₄", "q₁", "q₂"]
    dstrbs = mapreduce(w -> w', vcat, nominal_trajectory(workset).u)
    dstrb_plot = plot(range, vcat(dstrbs, dstrbs[end, :]'), label=permutedims(dstrb_labels), seriestype=:steppost)

    plt = plot(state_plot, error_plot, dstrb_plot, layout=(3, 1))
    display(plt)

    return plt
end

for i in 1:N
    if i > M
        IterativeLQR.circshift_trajectory!(workset, -1)
        H = M
        n = i - M
    else
        H = i
        n = 0
    end

    invΣq = diagm([1e2, 1e2])

    dyn!(xnew, x, w, k) = dynamics!(xnew, x, w, k, n)
    run(x, w, k) = running_cost(x, w, k, invΣq, n)
    fin(x, k) = final_cost(x, k, n)

    IterativeLQR.iLQR!(
        workset,
        dyn!, (fx, fu, x, w, k) -> dynamics_diff!(dyn!, fx, fu, x, w, k),
        run, (lx, lu, lxx, lxu, luu, x, w, k) -> running_cost_diff!(run, lx, lu, lxx, lxu, luu, x, w, k),
        fin, (Φx, Φxx, x, k) -> final_cost_diff!(fin, Φx, Φxx, x, k),
        verbose=false, plotting_callback=workset -> plotting_callback(workset, n), maxiter=5, N=H
    )
end

## horizon shift counter-action
IterativeLQR.circshift_trajectory!(workset, -(M + 1))
nominal_trajectory(workset).u .= circshift(deepcopy(nominal_trajectory(workset).u), 1)
display(plotting_callback(workset))
nominal_trajectory(workset).x[end]
