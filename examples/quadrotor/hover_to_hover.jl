using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory, active_trajectory
using QuadrotorODE
using MeshCatBenchmarkMechanisms

using LinearAlgebra
using ForwardDiff
using Plots
using DataFrames, CSV
using BenchmarkTools
using Infiltrator

# Quadrotor model
quadrotor = QuadrotorODE.System(9.81, 0.5, diagm([0.0023, 0.0023, 0.004]), 0.1750, 1.0, 0.0245)

# Horizon and timestep
T = 4
N = 400
h = T / N

# Target state
xₜ = vcat([0, 0, 2.0], [1, 0, 0, 0], zeros(3), zeros(3))
uₜ = quadrotor.m * quadrotor.g / 4 * ones(4)

# Initial state and inputs
θ₀ = 3 * pi / 4
x₀ = vcat([1, -1, 1.0], [cos(θ₀/2), sin(θ₀/2), 0, 0], zeros(3), zeros(3))
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
function running_cost(x, u, _)
    r, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]
    dr = r - xₜ[1:3]
    dq⃗ = q[2:4]
    du = u - uₜ
    return 1e1 * dr'dr + 1e1 * dq⃗'dq⃗ + v'v + ω'ω + du'du
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
S = [
    769.104 5.28962e-9 -9.77477e-10 -3.42266e-9 715.141 -2.39279e-9 241.915 1.89084e-9 -5.83008e-11 -1.93777e-11 4.34069 -1.06051e-10
    5.28962e-9 769.104 -4.98199e-11 -715.141 5.29192e-9 -3.27945e-9 2.28355e-9 241.915 1.73479e-11 -4.34069 3.07872e-11 -1.3701e-10
    -9.77477e-10 -4.98199e-11 518.129 -7.07492e-10 -1.20579e-9 4.63535e-10 -3.66676e-10 4.70971e-11 81.6384 -4.5972e-12 -7.04069e-12 2.5152e-11
    -3.42266e-9 -715.141 -7.07492e-10 3160.22 -1.36165e-8 5.338e-9 -2.71134e-9 -534.185 -1.62892e-10 19.956 -8.46541e-11 2.21343e-10
    715.141 5.29192e-9 -1.20579e-9 -1.36165e-8 3160.22 -6.48328e-10 534.185 2.74521e-9 -1.89806e-10 -8.42911e-11 19.956 -5.13592e-11
    -2.39279e-9 -3.27945e-9 4.63535e-10 5.338e-9 -6.48328e-10 719.578 -9.02046e-10 -1.4253e-9 9.44497e-11 3.30704e-11 -2.60711e-12 27.6493
    241.915 2.28355e-9 -3.66676e-10 -2.71134e-9 534.185 -9.02046e-10 147.189 9.57854e-10 -3.55125e-11 -1.61548e-11 3.29503 -4.24648e-11
    1.89084e-9 241.915 4.70971e-11 -534.185 2.74521e-9 -1.4253e-9 9.57854e-10 147.189 1.72244e-11 -3.29503 1.63675e-11 -5.96181e-11
    -5.83008e-11 1.73479e-11 81.6384 -1.62892e-10 -1.89806e-10 9.44497e-11 -3.55125e-11 1.72244e-11 41.4829 -1.04616e-12 -1.14007e-12 4.78914e-12
    -1.93777e-11 -4.34069 -4.5972e-12 19.956 -8.42911e-11 3.30704e-11 -1.61548e-11 -3.29503 -1.04616e-12 1.45232 -5.25134e-13 1.37353e-12
    4.34069 3.07872e-11 -7.04069e-12 -8.46541e-11 19.956 -2.60711e-12 3.29503 1.63675e-11 -1.14007e-12 -5.25134e-13 1.45232 -2.59839e-13
    -1.06051e-10 -1.3701e-10 2.5152e-11 2.21343e-10 -5.13592e-11 27.6493 -4.24648e-11 -5.96181e-11 4.78914e-12 1.37353e-12 -2.59839e-13 9.80969
]

function final_cost(x, _)
    dx = QuadrotorODE.state_difference(x, xₜ)
    return dx' * S * dx
end

final_cost(_, _) = 0

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
    stacked_derivatives=false, state_difference=QuadrotorODE.state_difference, regularization=:cost,
    verbose=true, logging=true, plotting_callback=plotting_callback
)

# Visualization
vis = (@isdefined vis) ? vis : Visualizer()
render(vis)

## quadrotor and target
MeshCatBenchmarkMechanisms.set_quadrotor!(vis, 2 * quadrotor.a, 0.12, 0.25)
MeshCatBenchmarkMechanisms.set_target!(vis, 0.12)

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
