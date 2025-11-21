# # Quadrotor Recovery Problem
# In this example we show how to optimize the recovery of a quadrotor in an MPC context. We use
# quaternions for the quadrotor's orientation which lie on a manifold in the state space. Therefore,
# in this example we show how to use the solver's capabilities to optimize directly on the manifold.

# ## Dependencies
# Two of the dependencies are not registered, namely
# [`QuadrotorODE.jl`](https://github.com/JurajLieskovsky/QuadrotorODE.jl) and 
# [`MeshCatBenchmarkMechanisms.jl`](https://github.com/JurajLieskovsky/MeshCatBenchmarkMechanisms.jl).
# However, if you if run the examples from a clone of the repository according to the [instructions]
# (@ref "Running Examples") in the docs, they load automatically.

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

# ## Quadrotor model
# We start by initializing a model of the quadrotor which stores its parameters.
# Although the code is unit-less, in practical terms we are using SI units throughout this example. 

quadrotor = QuadrotorODE.System(9.81, 0.5, diagm([0.0023, 0.0023, 0.004]), 0.1750, 1.0, 0.0245)

# ## Horizon and timestep
# We will be optimizing the trajectory on a horizon of T=2s which we discretize into 200 steps.

T = 2
N = 200
h = T / N

# ## Target
# we set the target state and input as the equilibrium

xₜ = vcat([0, 0, 1.0], [1, 0, 0, 0], zeros(3), zeros(3))
uₜ = quadrotor.m * quadrotor.g / 4 * ones(4)

# ## Initial state and inputs
# As the initial state we will use a significant rotation around the x-axis

θ₀ = 3 * pi / 4
x₀ = vcat([0, 0, 1.0], [cos(θ₀ / 2), sin(θ₀ / 2), 0, 0], zeros(3), zeros(3))

# And for the inital input use a small utility function that calculates the dot product between
# the z-axes of a global and local frame of the quadrotor

zRz(q⃗) = 1 - 2 * (q⃗[1]^2 + q⃗[2]^2)

# to produce an input that somewhat minimizes movement of the quadrotor in space

u₀(_) = zRz(x₀[5:7]) * uₜ

# ## Dynamics
# To discretize the continuous-time dynamics of the system, we use a fourth order Runge-Kutta method

"""RK4 integration with zero-order hold on u"""
function dynamics!(xnew, x, u, _)
    f1 = QuadrotorODE.dynamics(quadrotor, x, u)
    f2 = QuadrotorODE.dynamics(quadrotor, x + 0.5 * h * f1, u)
    f3 = QuadrotorODE.dynamics(quadrotor, x + 0.5 * h * f2, u)
    f4 = QuadrotorODE.dynamics(quadrotor, x + h * f3, u)
    xnew .= x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return nothing
end

# and `ForwardDiff.jl` to obtain its Jacobians

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

# Cost function
# As the running cost we can use a function that is quadratic in `x` and `u`

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

# this means that the final cost can be derived as the inifinite-horizon LQR value function. To calculate it, we use the the coordinate jacobian, which allows us to linearize the system's dynamics, as well as the running cost, on the state-space manifold

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

function final_cost(x, _)
    dx = QuadrotorODE.state_difference(x, xₜ)
    return dx' * S * dx
end

function final_cost_diff!(Φx, Φxx, x, k)
    H = DiffResults.DiffResult(0.0, (Φx, Φxx))
    ForwardDiff.hessian!(H, x_ -> final_cost(x_, k), x)

    return nothing
end

# ## Plotting callback
# During the optimization process we will plot the position and orientation of the quadrotor in
# space, as well as the inputs and cumulative cost.

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

# ## Optimization
# First we initialize the workset based on the length of the horizon and dimensions of the systems
# state and input vector

workset = IterativeLQR.Workset{Float64}(QuadrotorODE.nx, QuadrotorODE.nu, N, QuadrotorODE.nz)

# ### Warmstart
# We first run and optimization that stabilizes the quadrotor at the target equilibrium. This
# will essentially computes the gains of an infinite-horizon LQR which will drastically improve the
# convergence rate of the second optimization from the perturbed state

IterativeLQR.set_initial_state!(workset, xₜ)
IterativeLQR.set_nominal_inputs!(workset, [uₜ for _ in 1:N])

IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, regularization=:none,
    state_difference=QuadrotorODE.state_difference, coordinate_jacobian=QuadrotorODE.jacobian,
    verbose=true, plotting_callback=plotting_callback
)

# Note that as the system's state lies on a manifold we supply `state_difference` and
# `coordinate_jacobian` to optimize directly on the manifold.

# ### Recovery
# The second optimization is now run with `rollout = :partial` to re-use the gains from the previous
# optimization

IterativeLQR.set_initial_state!(workset, x₀)
warmstart || IterativeLQR.set_initial_inputs!(workset, [u₀(k) for k in 1:N])

IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, regularization=:none, rollout=:partial,
    state_difference=QuadrotorODE.state_difference, coordinate_jacobian=QuadrotorODE.jacobian,
    verbose=true, plotting_callback=plotting_callback,
)

# ## Visualization
# We can visualize the execution of the trajectory using `MeshCat.jl` for which we wrote a wrapper.

vis = (@isdefined vis) ? vis : Visualizer()
render(vis)

# First we initialize the quadrotor and target

MeshCatBenchmarkMechanisms.set_quadrotor!(vis, 2 * quadrotor.a, 0.07, 0.12)
MeshCatBenchmarkMechanisms.set_target!(vis, 0.07)

# and their initial configurations

MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, nominal_trajectory(workset).x[1])
MeshCatBenchmarkMechanisms.set_target_position!(vis, xₜ[1:3])

# Then we animate the recovery

anim = MeshCatBenchmarkMechanisms.Animation(vis, fps=1 / h)
for (i, x) in enumerate(nominal_trajectory(workset).x)
    atframe(anim, i) do
        MeshCatBenchmarkMechanisms.set_quadrotor_state!(vis, x)
    end
end
setanimation!(vis, anim, play=false);

# You can view the animation in your browser at the address given by running `> vis` in your REPL.
