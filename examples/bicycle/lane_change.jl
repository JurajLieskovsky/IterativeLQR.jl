using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory
using KinematicBicycleODE
using MeshCatBenchmarkMechanisms

using ForwardDiff, DiffResults
using Plots
using DataFrames, CSV
using Infiltrator
using BenchmarkTools
using LinearAlgebra
using MatrixEquations

# Bicycle model
bicycle = KinematicBicycleODE.Model(1.0, 3)

# Horizon and timestep
T = 5
N = 500
h = T / N

# Target state
vₜ = 130 / 3.6 # ms^-1 - target velocity
xₜ = [3.5, 0, vₜ, 0]
uₜ = zeros(KinematicBicycleODE.nu)

# Initial state and inputs
x₀ = [0, 0, vₜ, 0]
u₀(_) = zeros(KinematicBicycleODE.nu)

# Algorithm and regularization
algorithm = :ilqr
regularization = :cost

# Dynamics
#= """Explicit Euler method with zero-order hold on u"""
function dynamics!(xnew, x, u, _)
    xnew .= x + h * KinematicBicycleODE.f(bicycle, x, u)
    return nothing
end =#

"""RK4 integration with zero-order hold on u"""
function dynamics!(xnew, x, u, _)
    f1 = KinematicBicycleODE.f(bicycle, x, u)
    f2 = KinematicBicycleODE.f(bicycle, x + 0.5 * h * f1, u)
    f3 = KinematicBicycleODE.f(bicycle, x + 0.5 * h * f2, u)
    f4 = KinematicBicycleODE.f(bicycle, x + h * f3, u)
    xnew .= x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return nothing
end

function dynamics_diff!(∇f, x, u, k)
    nx = KinematicBicycleODE.nx
    nu = KinematicBicycleODE.nu

    @views ForwardDiff.jacobian!(
        ∇f,
        (xnew, arg) -> dynamics!(xnew, view(arg, 1:nx), view(arg, nx+1:nx+nu), k),
        zeros(nx),
        vcat(x, u)
    )

    return nothing
end

function dynamics_diff!(∇f, ∇2f, x, u, k)
    nx = KinematicBicycleODE.nx
    nu = KinematicBicycleODE.nu

    function stacked_dynamics(arg)
        xnew = zeros(eltype(arg), nx)
        dynamics!(xnew, view(arg, 1:nx), view(arg, nx+1:nx+nu), k)
        return xnew
    end

    ForwardDiff.jacobian!(∇2f, (jac, arg) -> ForwardDiff.jacobian!(jac, stacked_dynamics, arg), ∇f, vcat(x, u))

    return nothing
end

# Running cost
function lateral_acceleration(model, x)
    β = atan(model.lr / (model.lf + model.lr) * tan(x[4]))
    ψ̇ = x[3] / model.lr * sin(β)
    return x[3] * ψ̇
end

function running_cost(x, u, _)
    w_x = [1e2, 1e1, 1e2, 1e1]
    w_u = [1e-1, 1e4]
    w_l = 1e2

    dx = [x[1] - xₜ[1], cos(x[2] / 2), x[3] - xₜ[3], x[4] - xₜ[4]]

    ay = lateral_acceleration(bicycle, x)

    return sum(w_x .* dx .^ 2) + sum(w_u .* (u - uₜ) .^ 2) + w_l * ay^2
end

function running_cost_diff!(∇l, ∇2l, x, u, k)
    nx = KinematicBicycleODE.nx
    nu = KinematicBicycleODE.nu

    H = DiffResults.DiffResult(0.0, (∇l, ∇2l))

    @views ForwardDiff.hessian!(
        H,
        arg -> running_cost(view(arg, 1:nx), view(arg, nx+1:nx+nu), k),
        vcat(x, u)
    )

    return nothing
end

# Final cost
S, _ = begin
    nx = KinematicBicycleODE.nx
    nu = KinematicBicycleODE.nu

    ∇f = zeros(nx, nx + nu)
    dynamics_diff!(∇f, xₜ, uₜ, 0)

    A = ∇f[:, 1:nx]
    B = ∇f[:, nx+1:nx+nu]

    ∇l, ∇2l = zeros(nx + nu), zeros(nx + nu, nx + nu)
    running_cost_diff!(∇l, ∇2l, xₜ, uₜ, 0)

    Q = ∇2l[1:nx, 1:nx]
    P = ∇2l[1:nx, nx+1:nx+nu]
    R = ∇2l[nx+1:nx+nu, nx+1:nx+nu]

    MatrixEquations.ared(A, B, R, Q, P)
end

function final_cost(x, _)
    # dx = x - xₜ
    # return dx' * S * dx
    return 0
end

function final_cost_diff!(Φx, Φxx, x, k)
    H = DiffResults.DiffResult(0.0, (Φx, Φxx))
    ForwardDiff.hessian!(H, x -> final_cost(x, k), x)
    return nothing
end

function plotting_callback(workset)
    range = 0:workset.N

    states = mapreduce(x -> (x - xₜ)', vcat, nominal_trajectory(workset).x)
    state_labels = ["x₁" "x₂" "x₃"]
    position_plot = plot(range, states, label=state_labels)

    inputs = mapreduce(u -> u', vcat, nominal_trajectory(workset).u)
    input_plot = plot(range, vcat(inputs, inputs[end, :]'), label="u", seriestype=:steppost)

    lateral_accelerations = map(
        (x, u) -> lateral_acceleration(bicycle, x),
        nominal_trajectory(workset).x,
        eachrow(vcat(inputs, inputs[end, :]'))
    )
    lateral_acceleration_plot = plot(range, lateral_accelerations, label="ay")

    cost_plot = plot(range, cumsum(nominal_trajectory(workset).l), label="c", seriestype=:steppost)

    plt = plot(position_plot, lateral_acceleration_plot, input_plot, cost_plot, layout=(4, 1))
    display(plt)

    return plt
end

# Trajectory optimization
workset = IterativeLQR.Workset{Float64}(KinematicBicycleODE.nx, KinematicBicycleODE.nu, N)
IterativeLQR.set_initial_state!(workset, x₀)

IterativeLQR.set_initial_inputs!(workset, [u₀(k) for k in 1:N])
df = IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, algorithm=algorithm,
    regularization=regularization,
    verbose=true, logging=true, plotting_callback=plotting_callback
)

CSV.write("bicycle/results/bicycle-lane_change-$algorithm-μ-$regularization.csv", df)
