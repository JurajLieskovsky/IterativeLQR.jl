using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory
using RungeKutta
using CartPoleODE

using LinearAlgebra
using Plots
using Infiltrator

# dynamics of the adaptive system
function f(x, u, w, p)
    model = CartPoleODE.Model(9.81, p..., 0.2)
    tsit5 = RungeKutta.Tsit5()

    function dynamics!(dx, x, u)
        dx .= CartPoleODE.f(model, x, u)
    end

    return RungeKutta.f(tsit5, dynamics!, x, u, 1e-2) + w
end

function h(x, _, v, _)
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

## measurement
μv = zeros(2)
Σv = 1e-6 * I(2)

# random noise
noise(μ,Σ) = μ + sqrt.(diag(Σ)) .* (rand(length(μ)) .- 0.5)
# noise(μ, _) = zeros(length(μ))

# Reference trajectory
x = [zeros(CartPoleODE.nx) for _ in 1:N+1]
y = [zeros(2) for _ in 1:N+1]

x[1] .= x0
y[1] .= h(x0, u[1], μv, p)

for i in 1:N
    x[i+1] .= f(x[i], u[i], noise(μw, Σw), p)
    y[i+1] .= h(x[i+1], u[i], noise(μv, Σv), p)
end

# figure
xs = mapreduce(x -> x', vcat, x)
ys = mapreduce(y -> y', vcat, y)

plt = plot(layout=(2,1))
plot!(plt,xs, subplot=1)
plot!(plt,ys, subplot=2)

