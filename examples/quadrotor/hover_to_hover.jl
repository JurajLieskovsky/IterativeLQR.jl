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
using MatrixEquations

# Quadrotor model
quadrotor = QuadrotorODE.System(9.81, 0.5, diagm([0.0023, 0.0023, 0.004]), 0.1750, 1.0, 0.0245)

# Horizon and timestep
T = 2
N = 200
h = T / N

# Target state
xₜ = vcat([0, 0, 1.0], [1, 0, 0, 0], zeros(3), zeros(3))
uₜ = quadrotor.m * quadrotor.g / 4 * ones(4)

# Directional utility for thrust gravity compensation
# Calculates the dot product between the z-axes of a global and local frame
zRz(q⃗) = 1 - 2 * (q⃗[1]^2 + q⃗[2]^2)

# Initial state and inputs
θ₀ = 3 * pi / 4
x₀ = vcat([0, 0, 1.0], [cos(θ₀ / 2), sin(θ₀ / 2), 0, 0], zeros(3), zeros(3))
u₀(_) = zRz(x₀[5:7]) * uₜ

# Algorithm, regularization, and warmstart
algorithm = :ddp
regularization = (:arg,)
regularization_approach = :gmw
warmstart = true

# Dynamics
# function dynamics!(xnew, x, u, _)
#     xnew .= x + h * QuadrotorODE.dynamics(quadrotor, x, u)
#     QuadrotorODE.normalize_state!(xnew)
#     return nothing
# end

"""RK4 integration with zero-order hold on u"""
function dynamics!(xnew, x, u, _)
    f1 = QuadrotorODE.dynamics(quadrotor, x, u)
    f2 = QuadrotorODE.dynamics(quadrotor, x + 0.5 * h * f1, u)
    f3 = QuadrotorODE.dynamics(quadrotor, x + 0.5 * h * f2, u)
    f4 = QuadrotorODE.dynamics(quadrotor, x + h * f3, u)
    xnew .= x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return nothing
end

function dynamics_diff!(∇f, x, u, k)
    nx = QuadrotorODE.nx
    nu = QuadrotorODE.nu

    @views ForwardDiff.jacobian!(
        ∇f,
        (xnew, arg) -> dynamics!(xnew, view(arg, 1:nx), view(arg, nx+1:nx+nu), k),
        zeros(nx),
        vcat(x, u)
    )

    return nothing
end

function dynamics_diff!(∇f, ∇2f, x, u, k)
    nx = QuadrotorODE.nx
    nu = QuadrotorODE.nu

    function stacked_dynamics(arg)
        xnew = zeros(eltype(arg), nx)
        dynamics!(xnew, view(arg, 1:nx), view(arg, nx+1:nx+nu), k)
        return xnew
    end

    ForwardDiff.jacobian!(∇2f, (jac, arg) -> ForwardDiff.jacobian!(jac, stacked_dynamics, arg), ∇f, vcat(x, u))

    return nothing
end

# Running cost
function running_cost(x, u, _)
    r, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]
    q⃗ = q[2:4]
    dr = r - xₜ[1:3]
    du = u - uₜ
    return h * (dr'dr + q⃗'q⃗ / 4 + 1e-1 * v'v + 1e-1 * ω'ω + 1e-1 * du'du)
end

function running_cost_diff!(∇l, ∇2l, x, u, k)
    nx = QuadrotorODE.nx
    nu = QuadrotorODE.nu

    result = DiffResults.DiffResult(0.0, (∇l, ∇2l))
    @views ForwardDiff.hessian!(result, arg -> running_cost(view(arg, 1:nx), view(arg, nx+1:nx+nu), k), vcat(x, u))

    return nothing
end

# Final cost
## infinite horizion LQR value function's matrix
S, _ = begin
    nx = QuadrotorODE.nx
    nu = QuadrotorODE.nu

    ∇f, ∇l, ∇2l = zeros(nx, nx + nu), zeros(nx + nu), zeros(nx + nu, nx + nu)
    dynamics_diff!(∇f, xₜ, uₜ, 0)
    running_cost_diff!(∇l, ∇2l, xₜ, uₜ, 0)

    E = QuadrotorODE.jacobian(xₜ)

    A = E' * ∇f[:, 1:nx] * E
    B = E' * ∇f[:, nx+1:nx+nu]
    Q = 2 * E' * ∇2l[1:nx, 1:nx] * E
    R = 2 * ∇2l[nx+1:nx+nu, nx+1:nx+nu]

    MatrixEquations.ared(A, B, R, Q)
end

## resulting final cost and its partial derivatives
function final_cost(x, _)
    dx = QuadrotorODE.state_difference(x, xₜ)
    return dx' * S * dx
end

function final_cost_diff!(Φx, Φxx, x, k)
    H = DiffResults.DiffResult(0.0, (Φx, Φxx))
    ForwardDiff.hessian!(H, x_ -> final_cost(x_, k), x)

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

# Workset
workset = IterativeLQR.Workset{Float64}(QuadrotorODE.nx, QuadrotorODE.nu, N, QuadrotorODE.nz)

## warmstart by first creating a stabilizing controller at equilibrium
if warmstart
    IterativeLQR.set_initial_state!(workset, xₜ)
    IterativeLQR.set_initial_inputs!(workset, [uₜ for _ in 1:N])

    IterativeLQR.iLQR!(
        workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
        stacked_derivatives=true, algorithm=algorithm,
        regularization=regularization, regularization_approach=regularization_approach,
        state_difference=QuadrotorODE.state_difference,
        coordinate_jacobian=QuadrotorODE.jacobian,
        verbose=true, plotting_callback=plotting_callback
    )
end

## re-stabilization
IterativeLQR.set_initial_state!(workset, x₀)
warmstart || IterativeLQR.set_initial_inputs!(workset, [u₀(k) for k in 1:N])

df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, algorithm=algorithm, rollout=warmstart ? :partial : :full,
    regularization=regularization, regularization_approach=regularization_approach,
    state_difference=QuadrotorODE.state_difference,
    coordinate_jacobian=QuadrotorODE.jacobian,
    verbose=true, logging=true, plotting_callback=plotting_callback,
)


# Save iterations log to csv
warmstart_string = warmstart ? "-warmstart" : ""
regularization_string = isempty(regularization) ? "" : mapreduce(a -> "-$a", *, regularization)
CSV.write("quadrotor/results/quadrotor$warmstart_string-$algorithm$regularization_string-$regularization_approach.csv", df)

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
