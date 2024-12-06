using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory, active_trajectory, Workset
using RungeKutta
using CartPoleODE

using LinearAlgebra
using Plots
using Infiltrator
using ForwardDiff

# dynamics of the adaptive system
function f!(xnew, x, u, w, p)
    model = CartPoleODE.Model(9.81, p..., 0.2)
    tsit5 = RungeKutta.Tsit5()

    function dynamics!(dx, x, u)
        dx .= CartPoleODE.f(model, x, u)
    end

    RungeKutta.f!(xnew, tsit5, dynamics!, x, u, 1e-2)
    xnew .+= w

    return nothing
end

function h(x, v)
    return x[1:2] + v
end

# Horizon, initial state, and inputs
N = 300
x0 = [0, 3 / 4 * pi, 0, 0]
p_accurate = [1, 0.1]

# Noise models
## process
μw = zeros(CartPoleODE.nx)
Σw = diagm([1e-5, 1e-4, 1e-4, 1e-3])
invΣw = inv(Σw)

## measurement
μv = zeros(2)
Σv = 1e-6 * I(2)
invΣv = inv(Σv)

# Controller
controller(_) = zeros(CartPoleODE.nu)

# random noise
# noise(μ, Σ) = μ + sqrt.(diag(Σ)) .* (rand(length(μ)) .- 0.5)
noise(μ, _) = zeros(length(μ))

# Reference trajectory
x = [zeros(CartPoleODE.nx) for _ in 1:N+1]
z = [zeros(2) for _ in 1:N+1]
u = [zeros(CartPoleODE.nu) for _ in 1:N]

x[1] .= x0
z[1] .= h(x0, μv)

for i in 1:N
    # @infiltrate
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

function dynamics!(ynew, y, w, k, k0=0)
    @views begin
        xnew = ynew[1:4]
        pnew = ynew[5:6]

        x = y[1:4]
        p = y[5:6]

        wx = w[1:4]
        wp = w[5:6]
    end

    f!(xnew, x, u[k+k0], wx, p)
    pnew .= p + wp
end

function running_cost(y, w, k, k0=0)
    invΣwp = diagm([1e2, 1e4])

    @views begin
        x = y[1:4]

        wx = w[1:4]
        wp = w[5:6]
    end

    dz = z[k+k0] - h(x, μv)
    dw = μw - wx

    return 0.5 * (dz' * invΣv * dz + dw' * invΣw * dw + wp' * invΣwp * wp)
end

function final_cost(y, k, k0=0)
    x = y[1:4]
    dz = z[k+k0] - h(x, μv)
    return 0.5 * dz' * invΣv * dz
end

## plotting callback
function plotting_callback(workset, k0=0)
    range = 0:workset.N

    states = mapreduce(x -> x', vcat, nominal_trajectory(workset).x)
    state_labels = ["x₁", "x₂", "x₃", "x₄", "p₁", "p₂"]
    state_plot = plot(range, states, label=permutedims(state_labels))

    errors = mapreduce(
        (x_, x_ref_) -> (x_ - vcat(x_ref_, p_accurate))',
        vcat,
        nominal_trajectory(workset).x, circshift(x, -k0)
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
        n = M
        k0 = i - M
    else
        n = i
        k0 = 0
    end

    dyn!(xnew, x, w, k) = dynamics!(xnew, x, w, k, k0)
    run(x, w, k) = running_cost(x, w, k, k0)
    fin(x, k) = final_cost(x, k, k0)

    IterativeLQR.iLQR!(
        workset,
        dyn!, (fx, fu, x, w, k) -> dynamics_diff!(dyn!, fx, fu, x, w, k),
        run, (lx, lu, lxx, lxu, luu, x, w, k) -> running_cost_diff!(run, lx, lu, lxx, lxu, luu, x, w, k),
        fin, (Φx, Φxx, x, k) -> final_cost_diff!(fin, Φx, Φxx, x, k),
        verbose=false, plotting_callback=workset -> plotting_callback(workset, k0), maxiter=5, N=n
    )
end

## horizon shift counter-action
IterativeLQR.circshift_trajectory!(workset, -(M + 1))
nominal_trajectory(workset).u .= circshift(deepcopy(nominal_trajectory(workset).u), 1)
display(plotting_callback(workset))
nominal_trajectory(workset).x[end]

# Smooting
#=
initial_jt_state = vcat(x0, p0)
jt_noise_estimate = map(w -> vcat(w, zeros(2)), nominal_trajectory(workset).u)

jt_workset = IterativeLQR.Workset{Float64}(6, 6, N)
IterativeLQR.set_initial_state!(jt_workset, initial_jt_state)
IterativeLQR.set_initial_inputs!(jt_workset, jt_noise_estimate)

## dynamics, running cost, and final cost

## estimation
IterativeLQR.iLQR!(
    jt_workset,
    jt_dynamics!, (fx, fu, x, w, k) -> dynamics_diff!(jt_dynamics!, fx, fu, x, w, k),
    jt_running_cost, (lx, lu, lxx, lxu, luu, x, w, k) -> running_cost_diff!(jt_running_cost, lx, lu, lxx, lxu, luu, x, w, k),
    jt_final_cost, (Φx, Φxx, x, k) -> final_cost_diff!(jt_final_cost, Φx, Φxx, x, k),
    verbose=true, plotting_callback=jt_plotting_callback, maxiter=500, μ=1e2, μ_max=1e6
)

nominal_trajectory(jt_workset).x[end]
=#
