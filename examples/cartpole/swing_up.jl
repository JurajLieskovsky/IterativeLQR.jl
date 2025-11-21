# # Cart-Pole Swing-Up Problem
# In this example we show how to optimize a cart-pole swing-up with an MPC-like cost function.

# ## Dependencies
# Two of the dependencies are not registered, namely
# [CartPoleODE](https://github.com/JurajLieskovsky/CartPoleODE.jl) and 
# [MeshCatBenchmarkMechanisms](https://github.com/JurajLieskovsky/MeshCatBenchmarkMechanisms.jl).
# However, if you if run the examples from a clone of the repository according to the
# [instructions](@ref "Running Examples") in the docs, they load automatically.

using Revise

using CartPoleODE
using MeshCatBenchmarkMechanisms

using IterativeLQR
using IterativeLQR: nominal_trajectory

using ForwardDiff, DiffResults
using Plots
using DataFrames, CSV
using Infiltrator
using BenchmarkTools
using LinearAlgebra
using MatrixEquations

# ## Cart-pole model
# We start by initializing a model of the cart-pole which stores its parameters.
# Although the code is unit-less, in practical terms we are using SI units throughout this example. 

cartpole = CartPoleODE.Model(9.81, 1, 0.1, 0.2)

# ## Horizon and timestep
# We will be optimizing the trajectory on a horizon of T=2s which we discretize into 200 steps.

T = 2
N = 200
h = T / N

# ## Initial state and inputs
# For the intial state and nominal control policy we use the stable equilibrium

x₀ = [0.0, 0, 0, 0]

# and a simple harmonic input signal

u₀(k) = cos(2 * pi * (k - 1) / N - 1) * ones(CartPoleODE.nu)

# which perturbs the initial trajectory from stable equilibrium, where the gradient is zero.

# ## Regularization
# As the overall problem is highly non-convex, we use the eigen value regularization approach.

regularization = :eig

# ## Dynamics
# To discretize the continuous-time dynamics of the system, we use a fourth order Runge-Kutta method

"""RK4 integration with zero-order hold on u"""
function dynamics!(xnew, x, u, _)
    f1 = CartPoleODE.f(cartpole, x, u)
    f2 = CartPoleODE.f(cartpole, x + 0.5 * h * f1, u)
    f3 = CartPoleODE.f(cartpole, x + 0.5 * h * f2, u)
    f4 = CartPoleODE.f(cartpole, x + h * f3, u)
    xnew .= x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return nothing
end

# and `ForwardDiff.jl` to obtain its Jacobians

function dynamics_diff!(∇f, x, u, k)
    nx = CartPoleODE.nx
    nu = CartPoleODE.nu

    @views ForwardDiff.jacobian!(
        ∇f,
        (xnew, arg) -> dynamics!(xnew, view(arg, 1:nx), view(arg, nx+1:nx+nu), k),
        zeros(nx),
        vcat(x, u)
    )

    return nothing
end

# ## Cost functions
# We base the running and final cost on a distance function from the equlibrium

ξ(x) = [x[1], cos(x[2] / 2), x[3], x[4]]

# As `ξ` can also be regard as an alternative state (although the underlying representation
# has a singularity at the stable equilibrium) we may model the cost function after
# infinite-horizon LQR with weights

Q = h * diagm([1e1, 1e2, 1, 1])
R = h * Matrix{Float64}(I, 1, 1)

# The final cost can be formed from the solution of the DARE for the alternative state-space
# representation

S, _ = begin
    x_eq = [0.0, pi, 0, 0]
    u_eq = [0.0]

    nx = CartPoleODE.nx
    nu = CartPoleODE.nu

    ∇f = zeros(nx, nx + nu)
    dynamics_diff!(∇f, x_eq, u_eq, 0)

    E = ForwardDiff.jacobian(ξ, x_eq)
    A = E' * ∇f[:, 1:nx] * inv(E)
    B = E' * ∇f[:, nx+1:nx+nu]

    MatrixEquations.ared(A, B, R, Q)
end

# The running cost is then simply the quadratic form in `ξ` and `u`

running_cost(x, u, _) = ξ(x)' * Q * ξ(x) + u' * R * u

function running_cost_diff!(∇l, ∇2l, x, u, k)
    nx = CartPoleODE.nx
    nu = CartPoleODE.nu

    H = DiffResults.DiffResult(0.0, (∇l, ∇2l))

    @views ForwardDiff.hessian!(
        H,
        arg -> running_cost(view(arg, 1:nx), view(arg, nx+1:nx+nu), k),
        vcat(x, u)
    )

    return nothing
end

# And the final cost essentially a quadratic value function

final_cost(x, _) = ξ(x)' * S * ξ(x)

function final_cost_diff!(Φx, Φxx, x, k)
    H = DiffResults.DiffResult(0.0, (Φx, Φxx))
    ForwardDiff.hessian!(H, x -> final_cost(x, k), x)
    return nothing
end

# ## Plotting callback
# During the optimization process we will plot the position of the cart and angle of the pendulum
# as well as the input and cumulative cost.

function plotting_callback(workset)
    range = 0:workset.N

    states = mapreduce(x -> x', vcat, nominal_trajectory(workset).x)
    state_labels = ["x₁" "x₂" "x₃" "x₄"]
    position_plot = plot(range, states[:, 1:2], label=state_labels[1:1, 1:2])

    inputs = mapreduce(u -> u', vcat, nominal_trajectory(workset).u)
    input_plot = plot(range, vcat(inputs, inputs[end, :]'), label="u", seriestype=:steppost)

    cost_plot = plot(range, cumsum(nominal_trajectory(workset).l), label="c", seriestype=:steppost)

    plt = plot(position_plot, input_plot, cost_plot, layout=(3, 1))
    display(plt)

    return plt
end

# ## Trajectory optimization
# First we initialize the workset based on the length of the horizon and dimensions of the systems
# state and input vector

workset = IterativeLQR.Workset{Float64}(CartPoleODE.nx, CartPoleODE.nu, N)

# Then we set the initial state and nominal control policy

IterativeLQR.set_initial_state!(workset, x₀)
IterativeLQR.set_nominal_inputs!(workset, [u₀(k) for k in 1:N])

# Having done so, we run the optimization

IterativeLQR.iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!,
    stacked_derivatives=true, regularization=regularization,
    verbose=true, plotting_callback=plotting_callback
)

# ## Visualization
# We can visualize the execution of the trajectory using `MeshCat.jl` for which we wrote a wrapper.

(@isdefined vis) || (vis = Visualizer())
render(vis)

# First we initialize the cart-pole

MeshCatBenchmarkMechanisms.set_cartpole!(vis, 0.1, 0.05, 0.05, cartpole.l, 0.02)

# then set the initial configuration

MeshCatBenchmarkMechanisms.set_cartpole_state!(vis, nominal_trajectory(workset).x[1])

# and finally animate the swing-up

anim = MeshCatBenchmarkMechanisms.Animation(vis, fps=1 / h)
for (i, x) in enumerate(nominal_trajectory(workset).x)
    atframe(anim, i) do
        MeshCatBenchmarkMechanisms.set_cartpole_state!(vis, x)
    end
end
setanimation!(vis, anim, play=false)

# You can view the animation in your browser at the address given by running `> vis` in your REPL.
