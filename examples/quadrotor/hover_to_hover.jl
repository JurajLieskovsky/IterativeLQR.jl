using QuadrotorODE: state_difference
using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory, active_trajectory
using RungeKutta
using QuadrotorODE

using LinearAlgebra
using ForwardDiff
using Plots

# Horizon and timestep
T = 2
N = 200
h = T / N

# Equilibrium (target hovering) state and input
r₀ = zeros(3)
q₀ = [1, 0, 0, 0]
v₀ = zeros(3)
ω₀ = zeros(3)

x₀ = vcat(r₀, q₀, v₀, ω₀)
u₀ = 9.81 / 4 * ones(4)

# Dynamics
quadrotor = QuadrotorODE.System([0, 0, -9.81], 1, I(3), 0.3, 0.01)

rk4 = RungeKutta.RK4()

dynamics!(xnew, x, u) = RungeKutta.f!(
    xnew,
    rk4,
    (xnew, dz, u) -> xnew .= QuadrotorODE.forward_dynamics(quadrotor, x, u),
    x,
    u,
    h
)

function dynamics_diff!(fx, fu, x, u)
    f!(dznew, x₀, dz, u) = RungeKutta.f!(
        dznew,
        rk4,
        (dznew, dz, u) -> dznew .= QuadrotorODE.tangential_forward_dynamics(quadrotor, x₀, dz, u),
        dz,
        u,
        h
    )
    dz = zeros(12)
    fx .= ForwardDiff.jacobian((dznew_, dz_) -> f!(dznew_, x, dz_, u), zeros(12), dz)
    fu .= ForwardDiff.jacobian((dznew_, u_) -> f!(dznew_, x, dz, u_), zeros(12), u)
end

# Running cost
running_cost(_, u) = 1e-5 * h * u' * u

function running_cost_diff!(lx, lu, lxx, lxu, luu, x, u)
    ∇x!(grad, dx, u) = ForwardDiff.gradient!(grad, (dx_) -> running_cost(QuadrotorODE.incremented_state(x, dx_), u), dx)
    ∇u!(grad, dx, u) = ForwardDiff.gradient!(grad, (u_) -> running_cost(QuadrotorODE.incremented_state(x, dx), u_), u)

    dx = zeros(12)
    ForwardDiff.jacobian!(lxx, (grad, dx_) -> ∇x!(grad, dx_, u), lx, dx)
    ForwardDiff.jacobian!(lxu, (grad, u_) -> ∇x!(grad, dx, u_), lx, u)
    ForwardDiff.jacobian!(luu, (grad, u_) -> ∇u!(grad, dx, u_), lu, u)

    return nothing
end

# Final cost
function final_cost(x)
    dx = QuadrotorODE.state_difference(x, x₀)
    return dx' * diagm(vcat(1e1 * ones(6), 1e-2 * ones(6))) * dx
end

function final_cost_diff!(Φx, Φxx, x)
    dz = zeros(12)
    result = DiffResults.HessianResult(dz)
    ForwardDiff.hessian!(result, dz_ -> final_cost(QuadrotorODE.incremented_state(x, dz_)), dz)

    Φx .= result.derivs[1]
    Φxx .= result.derivs[2]

    return nothing
end

# Workset
workset = IterativeLQR.Workset{Float64}(13, 4, N, 12)
IterativeLQR.set_initial_state!(workset, vcat([-3, -3, -1], [cos(pi / 16), 0, 0, sin(pi / 16)], v₀, ω₀))
IterativeLQR.set_initial_inputs!(workset, [u₀ for _ in 1:N])

# Plotting callback
function plotting_callback(workset)
    N = workset.N

    states = mapreduce(x -> x', vcat, nominal_trajectory(workset).x)
    state_labels = ["x" "y" "z" "q₀" "q₁" "q₂" "q₃" "vx" "vy" "vz" "ωx" "ωy" "ωz"]
    position_plot = plot(0:N, states[:, 1:7], label=state_labels[1:1, 1:7])

    inputs = mapreduce(u -> u', vcat, nominal_trajectory(workset).u)
    input_labels = ["u₀" "u₁" "u₂" "u₃"]
    input_plot = plot(0:N-1, inputs, label=input_labels, seriestype=:steppost)

    cost_plot = plot(0:N, cumsum(nominal_trajectory(workset).l), label="c", seriestype=:steppost)

    plt = plot(position_plot, input_plot, cost_plot, layout=(3, 1))
    display(plt)

    return plt
end

IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    verbose=true, plotting_callback=plotting_callback, state_difference=QuadrotorODE.state_difference
)
