
include("CartPole.jl")
include("RungeKutta.jl")
using .CartPole
using .RungeKutta

using ForwardDiff
using Plots

# Runge-Kutta method
rk = RungeKutta.Method(
    [
        0 0 0 0 0 0
        1/4 0 0 0 0 0
        3/32 9/32 0 0 0 0
        1932/2197 -7200/2197 7296/2197 0 0 0
        439/216 -8 3680/513 -845/4104 0 0
        -8/27 2 3544/2565 1859/4104 -11/40 0
    ],
    [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55],
    [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
)

# Continuous-time dynamics
m = CartPole.Model(9.8, 1, 0.5, 0.1)
fc!(dx, x, u) = CartPole.f!(m, dx, x, u)

# Discrete-time dynamics
h = 1e-3
fd!(xnew,x,u) = RungeKutta.f!(xnew, rk, fc!, x, u, h)

# Simulation
N = 1000
xs = Matrix{Float64}(undef, 4, N + 1)

xs[:, 1] = [0, 3 * pi / 4, 0, 0]
for i in 1:N
    @views fd!(xs[:, i+1], xs[:, i], [0])
end

# Plotting
plt = plot()
plot!(plt, 1:N+1, xs[1, :])
plot!(plt, 1:N+1, xs[2, :])


#= function dF!(dFdx, dFdu, F!, x, u)
    nx = length(x)
    nu = length(u)

    @views stacked_F!(xnew, xu) = F!(xnew, xu[1:nx], xu[nx+1, nx+nu])

    ForwardDiff.jacobian!(stacked_F!)

end =#
