using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory, active_trajectory
using QuadrotorODE
using QuadrotorODE.Quaternions: rot, conjugate
using MeshCatBenchmarkMechanisms

using LinearAlgebra
using ForwardDiff
using Plots
using DataFrames, CSV
using BenchmarkTools
using Infiltrator
using MatrixEquations

# Quadrotor model
quadrotor = QuadrotorODE.System(9.81, 0.5, diagm([0.0023, 0.0023, 0.004]), 0.1750, 1.0, 0.0245)

# Horizon and timestep
T = 3
N = 300
h = T / N

# Target state
xₜ = vcat([0, 0, 1.0], [1, 0, 0, 0], zeros(3), zeros(3))
uₜ = quadrotor.m * quadrotor.g / 4 * ones(4)

# Initial state and inputs
θ₀ = 3 * pi / 4
x₀ = vcat([0, 0, 1.0], [cos(θ₀ / 2), sin(θ₀ / 2), 0, 0], zeros(3), zeros(3))
u₀ = uₜ
us₀ = [u₀ for _ in 1:N]

# Dynamics
function dynamics!(xnew, x, u, k, normalize=true)
    xnew .= x + h * QuadrotorODE.dynamics(quadrotor, x, u)
    normalize && QuadrotorODE.normalize_state!(xnew)
    return nothing
end

function dynamics_diff!(fx, fu, x, u, k)
    xnew = zeros(13)

    A = ForwardDiff.jacobian((xnew_, x_) -> dynamics!(xnew_, x_, u, k, false), xnew, x)
    B = ForwardDiff.jacobian((xnew_, u_) -> dynamics!(xnew_, x, u_, k, false), xnew, u)

    cE = QuadrotorODE.jacobian(x)
    nE = QuadrotorODE.jacobian(xnew)

    fx .= nE' * A * cE
    fu .= nE' * B

    return nothing
end

# Running cost
zRz(q⃗) = 1 - 2 * (q⃗[1]^2 + q⃗[2]^2) # k̂⋅R(q)k̂

function running_cost(x, u, _)
    r, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]
    q⃗ = q[2:4]
    dr = r - xₜ[1:3]
    du = u - zRz(q⃗) * uₜ
    return 1e1 * dr'dr + 1e1 * q⃗'q⃗ + v'v + ω'ω + du'du
end

function running_cost_diff!(lx, lu, lxx, lxu, luu, x, u, k)
    E = QuadrotorODE.jacobian(x)

    ∇x(x, u) = ForwardDiff.gradient(x_ -> running_cost(x_, u, k), x)
    ∇u(x, u) = ForwardDiff.gradient(u_ -> running_cost(x, u_, k), u)

    lx .= E' * ∇x(x, u)
    lu .= ∇u(x, u)
    lxx .= E' * ForwardDiff.jacobian(x_ -> ∇x(x_, u), x) * E
    lxu .= E' * ForwardDiff.jacobian(u_ -> ∇x(x, u_), u)
    luu .= ForwardDiff.jacobian(u_ -> ∇u(x, u_), u)

    return nothing
end

# Final cost
## Taylor expansions of the system's dynamics and running cost around equilibrium
fx_eq, fu_eq = zeros(12, 12), zeros(12, 4)
lx_eq, lu_eq, lxx_eq, lxu_eq, luu_eq = zeros(12), zeros(4), zeros(12, 12), zeros(12, 4), zeros(4, 4)

dynamics_diff!(fx_eq, fu_eq, xₜ, uₜ, 0)
running_cost_diff!(lx_eq, lu_eq, lxx_eq, lxu_eq, luu_eq, xₜ, uₜ, 0)

## check that running cost has a convex approximation and local minimum at equilibrium
@assert isposdef([lxx_eq lxu_eq; lxu_eq' luu_eq]) # more stict than necessary 
@assert all(isapprox.(lx_eq, 0))
@assert all(isapprox.(lu_eq, 0))

## value function's matrix
S, _ = ared(fx_eq, fu_eq, luu_eq, lxx_eq, lxu_eq)

## resulting final cost
function final_cost(x, _)
    dx = QuadrotorODE.state_difference(x, xₜ)
    return dx' * S * dx
end

function final_cost_diff!(Φx, Φxx, x, k)
    E = QuadrotorODE.jacobian(x)

    grad, hess = zeros(13), zeros(13, 13)
    H = DiffResults.DiffResult(0.0, (grad, hess))
    ForwardDiff.hessian!(H, x_ -> final_cost(x_, k), x)

    Φx .= E' * H.derivs[1]
    Φxx .= E' * H.derivs[2] * E

    return nothing
end

# Plotting callback
function plotting_callback(workset)
    range = 0:workset.N

    states = mapreduce(x -> x', vcat, nominal_trajectory(workset).x)
    state_labels = ["x" "y" "z" "q₀" "q₁" "q₂" "q₃" "vx" "vy" "vz" "ωx" "ωy" "ωz"]
    position_plot = plot(range, states[:, 1:7], label=state_labels[1:1, 1:7])

    inputs = mapreduce(u -> u', vcat, nominal_trajectory(workset).u)
    input_labels = ["u₀" "u₁" "u₂" "u₃"]
    input_plot = plot(range, vcat(inputs, inputs[end, :]'), label=input_labels, seriestype=:steppost)

    cost_plot = plot(range, cumsum(nominal_trajectory(workset).l), label="c", seriestype=:steppost)

    plt = plot(position_plot, input_plot, cost_plot, layout=(3, 1))
    display(plt)

    return plt
end

# Trajectory optimization
workset = IterativeLQR.Workset{Float64}(13, 4, N, 12)
IterativeLQR.set_initial_state!(workset, x₀)
IterativeLQR.set_initial_inputs!(workset, us₀)

df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=false, state_difference=QuadrotorODE.state_difference, regularization=:none,
    verbose=true, logging=true, plotting_callback=plotting_callback
)

# Visualization
vis = (@isdefined vis) ? vis : Visualizer()
render(vis)

## quadrotor and target
MeshCatBenchmarkMechanisms.set_quadrotor!(vis, 2 * quadrotor.a, 0.07, 0.12)
MeshCatBenchmarkMechanisms.set_target!(vis, 0.07)

## initial configuration
MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, nominal_trajectory(workset).x[1])
MeshCatBenchmarkMechanisms.set_target_position!(vis, xₜ[1:3])

## animation
anim = MeshCatBenchmarkMechanisms.Animation(vis, fps=1 / h)
for (i, x) in enumerate(nominal_trajectory(workset).x)
    atframe(anim, i) do
        MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, x)
    end
end
setanimation!(vis, anim, play=false);
