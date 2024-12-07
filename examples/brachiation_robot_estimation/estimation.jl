using Revise

using IterativeLQR
using IterativeLQR: nominal_trajectory, DataFrames
using RungeKutta
using BrachiationRobotODE

using LinearAlgebra
using Plots
using Infiltrator
using ForwardDiff
using DataFrames, CSV
using Interpolations

# Reference trajectory
## Raw data
df = DataFrame(CSV.File("brachiation_robot_estimation/results/csv/brachiation.csv"))
dropmissing!(df)

## Interpolation
tstep = 5e-3
df_interp = DataFrame(:time => 369.481:tstep:372) # 386

for col in names(df, Not(:time))
    interpolation = LinearInterpolation(df[!, :time], df[!, col])
    df_interp[!, col] = interpolation.(df_interp[!, :time])
end

df_interp[!, :theta_dot] .*= pi/180

z = [[row[col] for col in [:theta, :gamma, :theta_dot]] for row in eachrow(df_interp)]
u = [[row[:u]] for row in eachrow(df_interp)]

x0 = vcat(z[1], 0)
N = nrow(df_interp) - 1


# dynamics and measurements of the adaptive system
function f!(xnew, x, u, w, p)
    model = BrachiationRobotODE.Model(9.81, p[1], 0.25, 0.02, p[2], p[3], 0.02, 0.28, p[4], 0, 0)
    tsit5 = RungeKutta.Tsit5()
    RungeKutta.f!(xnew, tsit5, (ẋ_, x_, u_) -> BrachiationRobotODE.f!(model, ẋ_, x_, u_), x, u, tstep)
    xnew .+= w
    return nothing
end

function h(x, v)
    return x[1:3] + v
end

p0 = [0.67, 0.72, 2.5e-3, 0.1] # parameter guess

# Noise models
## process
μw = zeros(BrachiationRobotODE.nx)
Σw = diagm([1e-7, 1e-6, 1e-5, 1e-4])
invΣw = inv(Σw)

## measurement
μv = zeros(3)
Σv = diagm([2e-4, 1e-4, 5e-2])
invΣv = inv(Σv)

# Simulated trajectory
## Controller
setpoint(θ, θ̇) = pi / 2 * (1 + sign(sin(θ) * θ̇))

function controller(x)
    q, q̇ = x[1:2], x[3:4]
    P, D = 1.5, 0.1
    γ_des = setpoint(q[1], q̇[1])
    return -P * (q[2] - γ_des) - D * q̇[2]
end

## random noise
noise(μ, Σ) = μ + sqrt.(diag(Σ)) .* (rand(length(μ)) .- 0.5)
# noise(μ, _) = zeros(length(μ))

## trajectory
x_sim = [zeros(BrachiationRobotODE.nx) for _ in 1:N+1]
z_sim = [zeros(3) for _ in 1:N+1]
u_sim = [zeros(BrachiationRobotODE.nu) for _ in 1:N]

x_sim[1] .= x0
z_sim[1] .= h(x0, μv)

for i in 1:N
    u_sim[i] .= controller(x_sim[i])
    f!(x_sim[i+1], x_sim[i], u_sim[i], noise(μw, Σw), p0)
    z_sim[i+1] .= h(x_sim[i+1], noise(μv, Σv))
end

# display trajectory comparison and wait for user input to continue
data_plot = plot(layout=(2,1))

output_labels = ["z₁" "z₂" "z₃"]
plot!(data_plot, mapreduce(z_ -> z_', vcat, z), label=output_labels, subplot=1)
plot!(data_plot, mapreduce(z_ -> z_', vcat, z_sim), label=output_labels .* "_sim", subplot=1)

plot!(data_plot, mapreduce(u_ -> u_', vcat, u), label="u", subplot=2)
plot!(data_plot, mapreduce(u_ -> u_', vcat, u_sim), label="u_sim", subplot=2)

display(data_plot)
println("Press any key to continue...")
read(stdin, Char)

# Differentiation functions
function dynamics_diff!(dynamics!, fx, fu, x, w, k)
    f = similar(x)

    ForwardDiff.jacobian!(fx, (xnew, x_) -> dynamics!(xnew, x_, w, k), f, x)
    ForwardDiff.jacobian!(fu, (xnew, w_) -> dynamics!(xnew, x, w_, k), f, w)

    return nothing
end

function running_cost_diff!(running_cost, lx, lu, lxx, lxu, luu, x, w, k)
    ∇xL!(grad, x0, w0) = ForwardDiff.gradient!(grad, (x_) -> running_cost(x_, w0, k), x0)
    ∇uL!(grad, x0, w0) = ForwardDiff.gradient!(grad, (w_) -> running_cost(x0, w_, k), w0)

    ForwardDiff.jacobian!(lxx, (grad, x_) -> ∇xL!(grad, x_, w), lx, x)
    ForwardDiff.jacobian!(lxu, (grad, w_) -> ∇xL!(grad, x, w_), lx, w)
    ForwardDiff.jacobian!(luu, (grad, w_) -> ∇uL!(grad, x, w_), lu, w)

    return nothing
end

function final_cost_diff!(final_cost, Φx, Φxx, x, k)
    result = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(result, x -> final_cost(x, k), x)

    Φx .= result.derivs[1]
    Φxx .= result.derivs[2]

    return nothing
end

# Moving horizon estimation
initial_state = vcat(x0, p0)
noise_estimate = [vcat(μw, zeros(4)) for _ in 1:N]

workset = IterativeLQR.Workset{Float64}(8, 8, N)
IterativeLQR.set_initial_state!(workset, initial_state)
IterativeLQR.set_initial_inputs!(workset, noise_estimate)

M = 40 # observation horizon

function dynamics!(x̂new, x̂, û, k, n=0)
    @views begin
        xnew = x̂new[1:4]
        pnew = x̂new[5:8]

        x = x̂[1:4]
        p = x̂[5:8]
        w = û[1:4]
        q = û[5:8]
    end

    f!(xnew, x, u[k+n], w, p)
    pnew .= p + q
end

function running_cost(x̂, û, k, invΣq, n=0)
    @views begin
        x = x̂[1:4]
        w = û[1:4]
        q = û[5:8]
    end

    dz = z[k+n] - h(x, μv)
    dw = μw - w

    return 0.5 * (dz' * invΣv * dz + dw' * invΣw * dw + q' * invΣq * q)
end

function final_cost(x̂, k, n=0)
    x = x̂[1:4]
    dz = z[k+n] - h(x, μv)
    return 0.5 * dz' * invΣv * dz
end

## plotting callback
function plotting_callback(workset, n=0)
    range = 0:workset.N

    states = mapreduce(x -> x', vcat, nominal_trajectory(workset).x)
    state_labels = ["x₁", "x₂", "x₃", "x₄", "p₁", "p₂", "p₃", "p₄"]
    state_plot = plot(range, states, label=permutedims(state_labels))

    errors = mapreduce(
        (x_, z_) -> (h(x_, μv) - z_)',
        vcat,
        nominal_trajectory(workset).x, circshift(z, -n)
    )
    error_labels = ["Δz₁", "Δz₂", "Δz₃"]
    error_plot = plot(range, errors, label=permutedims(error_labels))

    dstrb_labels = ["w₁", "w₂", "w₃", "w₄", "q₁", "q₂", "q₃", "q₄"]
    dstrbs = mapreduce(w -> w', vcat, nominal_trajectory(workset).u)
    dstrb_plot = plot(range, vcat(dstrbs, dstrbs[end, :]'), label=permutedims(dstrb_labels), seriestype=:steppost)

    plt = plot(state_plot, error_plot, dstrb_plot, layout=(3, 1))
    display(plt)

    return plt
end

for i in 1:N
    if i > M
        IterativeLQR.circshift_trajectory!(workset, -1)
        H = M
        n = i - M
    else
        H = i
        n = 0
    end

    invΣq = diagm([1e6, 1e6, 1e8, 1e7])

    dyn!(xnew, x, w, k) = dynamics!(xnew, x, w, k, n)
    run(x, w, k) = running_cost(x, w, k, invΣq, n)
    fin(x, k) = final_cost(x, k, n)

    IterativeLQR.iLQR!(
        workset,
        dyn!, (fx, fu, x, w, k) -> dynamics_diff!(dyn!, fx, fu, x, w, k),
        run, (lx, lu, lxx, lxu, luu, x, w, k) -> running_cost_diff!(run, lx, lu, lxx, lxu, luu, x, w, k),
        fin, (Φx, Φxx, x, k) -> final_cost_diff!(fin, Φx, Φxx, x, k),
        verbose=true, plotting_callback=workset -> plotting_callback(workset, n), maxiter=5, N=H
    )
end

## horizon shift counter-action
IterativeLQR.circshift_trajectory!(workset, -(M + 1))
nominal_trajectory(workset).u .= circshift(deepcopy(nominal_trajectory(workset).u), 1)
display(plotting_callback(workset))
nominal_trajectory(workset).x[end]

# Post simulation
println("Press any key to continue...")
read(stdin, Char)

pf = nominal_trajectory(workset).x[end][5:8]

x_sim[1] .= x0
z_sim[1] .= h(x0, μv)

for i in 1:N
    u_sim[i] .= controller(x_sim[i])
    f!(x_sim[i+1], x_sim[i], u_sim[i], noise(μw, Σw), pf)
    z_sim[i+1] .= h(x_sim[i+1], noise(μv, Σv))
end

# display trajectory comparison and wait for user input to continue
data_plot = plot(layout=(2,1))

output_labels = ["z₁" "z₂" "z₃"]
plot!(data_plot, mapreduce(z_ -> z_', vcat, z), label=output_labels, subplot=1)
plot!(data_plot, mapreduce(z_ -> z_', vcat, z_sim), label=output_labels .* "_sim", subplot=1)

plot!(data_plot, mapreduce(u_ -> u_', vcat, u), label="u", subplot=2)
plot!(data_plot, mapreduce(u_ -> u_', vcat, u_sim), label="u_sim", subplot=2)

display(data_plot)
