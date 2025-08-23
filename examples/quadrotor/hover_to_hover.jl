using QuadrotorODE: state_difference
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

# Algorithm, regularization, and warmstart
algorithm = :ddp
regularization = :none # !!! cost regularization in tangent space isn't properly implemented yet
warmstart = false

# Dynamics
function dynamics!(xnew, x, u, k, normalize=true)
    xnew .= x + h * QuadrotorODE.dynamics(quadrotor, x, u)
    normalize && QuadrotorODE.normalize_state!(xnew)
    return nothing
end

function dynamics_diff!(∇f, x, u, k)
    nx = QuadrotorODE.nx
    nu = QuadrotorODE.nu

    @views ForwardDiff.jacobian!(
        ∇f,
        (xnew, arg) -> dynamics!(xnew, view(arg, 1:nx), view(arg, nx+1:nx+nu), k, false),
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
zRz(q⃗) = 1 - 2 * (q⃗[1]^2 + q⃗[2]^2) # k̂⋅R(q)k̂

function running_cost(x, u, _)
    r, q, v, ω = x[1:3], x[4:7], x[8:10], x[11:13]
    q⃗ = q[2:4]
    dr = r - xₜ[1:3]
    du = u - uₜ # change to `du = u - zRz(q⃗) * uₜ` after I figure out the regularization
    return dr'dr + q⃗'q⃗ + 1e-1 * v'v + 1e-1 * ω'ω + 1e-1 * du'du
end

function running_cost_diff!(∇l, ∇2l, x, u, k)
    nx = QuadrotorODE.nx
    nu = QuadrotorODE.nu

    result = DiffResults.DiffResult(0.0, (∇l, ∇2l))
    @views ForwardDiff.hessian!(result, arg -> running_cost(view(arg, 1:nx), view(arg, nx+1:nx+nu), k), vcat(x, u))

    return nothing
end

# Final cost
## Taylor expansions of the system's dynamics and running cost around equilibrium
∇f, ∇l, ∇2l = zeros(13, 17), zeros(17), zeros(17, 17)

dynamics_diff!(∇f, xₜ, uₜ, 0)
running_cost_diff!(∇l, ∇2l, xₜ, uₜ, 0)

## augmented forms
function augmented_coordinate_jacobian(x)
    nx = QuadrotorODE.nx
    nz = QuadrotorODE.nz
    nu = QuadrotorODE.nu

    aug_E = zeros(eltype(x), nx + nu, nz + nu)
    aug_E[1:nx, 1:nz] .= QuadrotorODE.jacobian(x)
    aug_E[nx+1:nx+nu, nz+1:nz+nu] .= Matrix{Float64}(I, nu, nu)
    return aug_E
end

aug_E = augmented_coordinate_jacobian(xₜ)

aug_∇f = QuadrotorODE.jacobian(xₜ)' * ∇f * aug_E
aug_∇l = aug_E' * ∇l
aug_∇2l = aug_E' * ∇2l * aug_E

## Check that first and second order conditions for local minima are satisfied
## This also asserts convexity
@assert isposdef(aug_∇2l)
@assert all(isapprox.(aug_∇l, 0))

## value function's matrix
K, S = begin
    A = aug_∇f[:, 1:12]
    B = aug_∇f[:, 13:16]
    R = aug_∇2l[13:16, 13:16]
    Q = aug_∇2l[1:12, 1:12]
    M = aug_∇2l[1:12, 13:16]

    S, _ = ared(A, B, R, Q, M)
    K = inv(R + B' * S * B) * B' * S * A

    K, S
end

## resulting final cost
function final_cost(x, _)
    dx = QuadrotorODE.state_difference(x, xₜ, :rp)
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

# Workset
workset = IterativeLQR.Workset{Float64}(QuadrotorODE.nx, QuadrotorODE.nu, N, QuadrotorODE.nz)

## warmstart by first creating a stabilizing controller at equilibrium
if warmstart
    IterativeLQR.set_initial_state!(workset, xₜ)
    IterativeLQR.set_initial_inputs!(workset, [uₜ for _ in 1:N])

    IterativeLQR.iLQR!(
        workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
        stacked_derivatives=true,
        state_difference=(x, xref) -> QuadrotorODE.state_difference(x, xref, :rp),
        coordinate_jacobian=QuadrotorODE.jacobian,
        regularization=regularization, algorithm=algorithm,
        verbose=true, plotting_callback=plotting_callback
    )
end

## re-stabilization
IterativeLQR.set_initial_state!(workset, x₀)
warmstart || IterativeLQR.set_initial_inputs!(workset, [u₀ for _ in 1:N])

df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, rollout=:partial,
    state_difference=(x, xref) -> QuadrotorODE.state_difference(x, xref, :rp),
    coordinate_jacobian=QuadrotorODE.jacobian,
    regularization=regularization, algorithm=algorithm,
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
