using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory
using RungeKutta
using CartPoleODE

using LinearAlgebra
using Plots
using Infiltrator

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
N = 100
x0 = [0, 3 / 4 * pi, 0, 0]
u = [zeros(CartPoleODE.nu) for _ in 1:N]
p = [1, 0.1]

# Noise models
## process
μw = zeros(CartPoleODE.nx)
Σw = diagm([1e-4, 1e-2, 1e-2, 1e-1])
invΣw = inv(Σw)

## measurement
μv = zeros(2)
Σv = 1e-6 * I(2)
invΣv = inv(Σv)

# random noise
noise(μ, Σ) = μ + sqrt.(diag(Σ)) .* (rand(length(μ)) .- 0.5)
# noise(μ, _) = zeros(length(μ))

# Reference trajectory
x = [zeros(CartPoleODE.nx) for _ in 1:N+1]
z = [zeros(2) for _ in 1:N+1]

x[1] .= x0
z[1] .= h(x0, μv)

for i in 1:N
    f!(x[i+1], x[i], u[i], noise(μw, Σw), p)
    z[i+1] .= h(x[i+1], noise(μv, Σv))
end

# figure
xs = mapreduce(x_ -> x_', vcat, x)
zs = mapreduce(z_ -> z_', vcat, z)

plt = plot(layout=(2, 1))
plot!(plt, xs, subplot=1)
plot!(plt, zs, subplot=2)
display(plt)

# MHE functions
dynamics!(xnew, x, w, k) = f!(xnew, x, u[k], w, p)

function running_cost(x, w, k)
    dy = z[k] - h(x, μv)
    dw = μw - w
    return 0.5 * (dy' * invΣv * dy + dw' * invΣw * dw)
end

function final_cost(x, k)
    dy = z[k] - h(x, μv)
    return 0.5 * dy' * invΣv * dy
end

# optimal estimation
workset = [IterativeLQR.Workset{Float64}(4, 4, n) for n in 1:N] # here nu is actually nw

for i in 1:N
    IterativeLQR.set_initial_state!(workset[i], x0)

    # copy noise estimates from previous iteration and add mean
    if i == 1
        IterativeLQR.set_initial_inputs!(workset[i], [μw])
    else
        IterativeLQR.set_initial_inputs!(workset[i], vcat(nominal_trajectory(workset[i-1]).u, [μw]))
    end

    # IterativeLQR.iLQR!(
    #     workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    #     verbose=true, logging=true, plotting_callback=plotting_callback, maxiter=200
    # )
end
