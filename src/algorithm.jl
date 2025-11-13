"""
Rolls-out the states x of the nominal trajectory from the initial state x̃ and nominal inputs u.
It also evaluates the running costs l in multiple threads and the final cost Φ. 

"""
function trajectory_rollout!(
    workset::Workset, dynamics!::Function, running_cost::Function, final_cost::Function
)
    @unpack N = workset
    @unpack x, u, l = nominal_trajectory(workset)

    @inbounds for k in 1:N
        dynamics!(x[k+1], x[k], u[k], k)
    end

    @threads for k in 1:N
        l[k] = running_cost(x[k], u[k], k)
    end

    l[N+1] = final_cost(x[N+1], N + 1)

    return true, sum(l)
end

"""
Produces partial derivatives of the system's dynamics f and costs l and Φ
that are required for the backward pass of the algorithm. (multithreaded)

"""
function differentiation!(
    workset::Workset,
    dynamics_diff!::Function, running_cost_diff!::Function, final_cost_diff!::Function,
    stacked::Bool
)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack ∇f, fx, fu = workset.dynamics_derivatives
    @unpack ∇l, lx, lu, ∇2l, lxx, lux, lxu, luu = workset.cost_derivatives
    @unpack Φx, Φxx = workset.cost_derivatives

    @threads for k in 1:N
        if stacked
            dynamics_diff!(∇f[k], x[k], u[k], k)
            running_cost_diff!(∇l[k], ∇2l[k], x[k], u[k], k)
        else
            dynamics_diff!(fx[k], fu[k], x[k], u[k], k)
            running_cost_diff!(lx[k], lu[k], lxx[k], lxu[k], luu[k], x[k], u[k], k)
            lux[k] .= lxu[k]'
        end
    end

    final_cost_diff!(Φx, Φxx, x[N+1], N + 1)

    return nothing
end

"""
Produces partial derivatives of the system's dynamics f and costs l and Φ  
that are required for the backward pass of the algorithm. It also transforms them  
to the surface of the state manifold using coordinate jacobians. (multithreaded)

"""
function differentiation!(
    workset::Workset,
    dynamics_diff!::Function, running_cost_diff!::Function, final_cost_diff!::Function,
    coordinate_jacobian::Function,
    stacked::Bool
)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack E = workset.coordinate_jacobians
    @unpack ∇f, fx, fu = workset.dynamics_derivatives
    @unpack ∇l, lx, lu, ∇2l, lxx, lux, lxu, luu = workset.cost_derivatives
    @unpack Φx, Φxx = workset.cost_derivatives

    dfwork = workset.dynamics_derivatives_workset
    dlwork = workset.cost_derivatives_workset

    # Coordinate jacobians (precalculated as E[k] and E[k+1] are required for dynamics derivs)
    @threads for k in 1:N+1
        E[k] .= coordinate_jacobian(x[k])
    end

    # Dynamics and running cost
    @threads for k in 1:N
        i = threadid()

        # differentiation
        if stacked
            dynamics_diff!(dfwork.∇f[i], x[k], u[k], k)
            running_cost_diff!(dlwork.∇l[i], dlwork.∇2l[i], x[k], u[k], k)
        else
            dynamics_diff!(dfwork.fx[i], dfwork.fu[i], x[k], u[k], k)
            running_cost_diff!(
                dlwork.lx[i], dlwork.lu[i], dlwork.lxx[i], dlwork.lxu[i], dlwork.luu[i], x[k], u[k], k
            )
        end

        # dynamics coordinate transformation
        fx[k] .= E[k+1]' * dfwork.fx[i] * E[k]
        fu[k] .= E[k+1]' * dfwork.fu[i]

        # running cost coordinate transformation
        lx[k] .= E[k]' * dlwork.lx[i]
        lu[k] .= dlwork.lu[i]

        lxx[k] .= E[k]' * dlwork.lxx[i] * E[k]
        lxu[k] .= E[k]' * dlwork.lxu[i]
        luu[k] .= dlwork.luu[i]
        lux[k] .= lxu[k]'
    end

    final_cost_diff!(dlwork.Φx, dlwork.Φxx, x[N+1], N + 1)

    Φx .= E[N+1]' * dlwork.Φx
    Φxx .= E[N+1]' * dlwork.Φxx * E[N+1]

    return nothing
end

"""
Regularizes the hessians of the running cost l and final cost Φ. (multithreaded)

"""
function cost_regularization!(workset::Workset, δ::Real, regularization::Symbol)
    @unpack N, nx, ndx = workset
    @unpack ∇2l, Φxx = workset.cost_derivatives

    @threads for k in 1:N
        regularize!(∇2l[k], δ, regularization)
    end

    regularize!(Φxx, δ, regularization)

    return nothing
end

"""
Performs the backward pass of the algorithm. This produces the the terms d and K
of the policy update as well as the expected improvement Δv.

"""
function backward_pass!(workset::Workset)
    @unpack N, nx, ndx, nu = workset
    @unpack d, K = workset.policy_update
    @unpack vx, vxx = workset.backward_pass_workset
    @unpack g, qx, qu = workset.backward_pass_workset
    @unpack H, qxx, quu, qux = workset.backward_pass_workset

    @unpack ∇f = workset.dynamics_derivatives
    @unpack ∇l, ∇2l = workset.cost_derivatives
    @unpack Φx, Φxx = workset.cost_derivatives

    Δv = 0
    vx .= Φx
    vxx .= Φxx

    @inbounds for k in N:-1:1
        # gradient and hessian of the argument
        g .= ∇l[k] + ∇f[k]' * vx
        H .= ∇2l[k] + ∇f[k]' * vxx * ∇f[k]

        # control update
        F = cholesky(Symmetric(quu))
        d[k] = -(F \ qu)
        K[k] = -(F \ qux)

        # cost-to-go model
        vx .= qx + K[k]' * qu
        vxx .= qxx + K[k]' * qux

        # expected improvement
        Δv -= 0.5 * d[k]' * quu * d[k]
    end

    d_∞ = mapreduce(d_k -> mapreduce(abs, max, d_k), max, d)
    d_2 = sqrt(mapreduce(d_k -> d_k'd_k, +, d))

    return Δv, d_∞, d_2
end

"""
Performs the forward pass of the algorithm. This produces a candidate trajectory x, u.

"""
function forward_pass!(
    workset::Workset,
    dynamics!::Function, difference::Function, running_cost::Function, final_cost::Function,
    α::Real
)
    @unpack N = workset
    @unpack x, u, l = active_trajectory(workset)
    @unpack d, K = workset.policy_update

    x_ref = nominal_trajectory(workset).x
    u_ref = nominal_trajectory(workset).u
    l_ref = nominal_trajectory(workset).l

    x[1] = x_ref[1]

    @inbounds for k in 1:N
        u[k] .= u_ref[k] + α * d[k] + K[k] * difference(x[k], x_ref[k])
        dynamics!(x[k+1], x[k], u[k], k)
    end

    @threads for k in 1:N
        l[k] = running_cost(x[k], u[k], k)
    end

    l[N+1] = final_cost(x[N+1], N + 1)

    return true, sum(l), sum(l) - sum(l_ref)
end

"""
Prints information about the current iteration.

"""
function print_iteration!(line_count, i, α, J, ΔJ, Δv, d_inf, d_2, accepted, diff, reg, bwd, fwd)
    line_count[] % 10 == 0 && @printf(
        "%-9s %-9s %-11s %-9s %-9s %-9s %-9s %-9s %-9s %-8s %-8s %-8s\n",
        "iter", "α", "J", "ΔJ", "ΔV", "d∞", "d2", "accepted", "diff", "reg", "bwd", "fwd"
    )
    @printf(
        "%-9i %-9.3g %-11.5g %-9.3g %-9.3g %-9.3g %-9.3g %-9s %-9.3g %-8.2g %-8.2g %-8.2g\n",
        i, α, J, ΔJ, Δv, d_inf, d_2, accepted, diff, reg, bwd, fwd
    )
    line_count[] += 1
end

"""
Creates a data frame for logging information about the iterations of the algorithm.

"""
iteration_dataframe() = DataFrame(
    i=Int[], α=Float64[], J=Float64[], ΔJ=Float64[], ΔV=Float64[], d_inf=Float64[], d_2=Float64[], accepted=Bool[],
    diff=Float64[], reg=Float64[], bwd=Float64[], fwd=Float64[]
)

"""
Logs information into the data frame about the iterations of the algorithm.

"""
function log_iteration!(dataframe, i, α, J, ΔJ, Δv, d_inf, d_2, accepted, diff, reg, bwd, fwd)
    push!(dataframe, (i, α, J, ΔJ, Δv, d_inf, d_2, accepted, diff, reg, bwd, fwd))
end

"""
Optimizes the trajectory of the controlled system.

"""
function iLQR!(
    workset::Workset,
    dynamics!::Function, dynamics_diff!::Function,
    running_cost::Function, running_cost_diff!::Function,
    final_cost::Function, final_cost_diff!::Function;
    maxiter::Int=250, ρ=1e-4, δ=sqrt(eps()), α_values=exp2.(0:-1:-16), termination_threshold=1e-4,
    rollout::Symbol=:full, verbose::Bool=true, logging::Bool=false, plotting_callback::Union{Function,Nothing}=nothing,
    stacked_derivatives::Bool=false, state_difference::Function=-, coordinate_jacobian::Union{Function,Nothing}=nothing,
    regularization::Symbol=:mchol
)
    @assert workset.ndx == workset.nx || coordinate_jacobian !== nothing

    # line count for printing
    line_count = Ref(0)

    # dataframe for logging
    dataframe = logging ? iteration_dataframe() : nothing

    # initial trajectory rollout
    if rollout == :full
        rlt = @elapsed begin
            successful, J = try
                trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
            catch
                false, NaN
            end
        end

        verbose && print_iteration!(line_count, 0, NaN, J, NaN, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)
        logging && log_iteration!(dataframe, 0, NaN, J, NaN, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)

        if !successful
            return nothing
        end
    elseif rollout == :partial
        rlt = @elapsed begin
            successful, J, ΔJ = try
                forward_pass!(workset, dynamics!, state_difference, running_cost, final_cost, 0)
            catch
                false, NaN, NaN
            end
        end

        verbose && print_iteration!(line_count, 0, NaN, J, ΔJ, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)
        logging && log_iteration!(dataframe, 0, NaN, J, ΔJ, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)

        if successful
            swap_trajectories!(workset)
        else
            return nothing
        end
    end

    (plotting_callback === nothing) || plotting_callback(workset)

    # algorithm
    for i in 1:maxiter
        # nominal trajectory differentiation
        diff = @elapsed begin
            if workset.ndx == workset.nx
                differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!, stacked_derivatives)
            else
                differentiation!(
                    workset, dynamics_diff!, running_cost_diff!, final_cost_diff!, coordinate_jacobian, stacked_derivatives
                )
            end
        end

        # regularization
        reg = regularization != :none ? @elapsed(cost_regularization!(workset, δ, regularization)) : NaN

        # backward pass
        bwd = @elapsed begin
            Δv1, d_∞, d_2 = backward_pass!(workset)
        end

        # forward pass
        accepted = false

        for α in α_values
            fwd = @elapsed begin
                successful, J, ΔJ = try
                    forward_pass!(workset, dynamics!, state_difference, running_cost, final_cost, α)
                catch
                    false, NaN, NaN
                end
            end

            # expected improvement and success evaluation
            Δv = (2 * α - α^2) * Δv1
            accepted = successful && ΔJ <= ρ * Δv

            # printout
            verbose && print_iteration!(line_count, i, α, J, ΔJ, Δv, d_∞, d_2, accepted, diff * 1e3, reg * 1e3, bwd * 1e3, fwd * 1e3)
            logging && log_iteration!(dataframe, i, α, J, ΔJ, Δv, d_∞, d_2, accepted, diff * 1e3, reg * 1e3, bwd * 1e3, fwd * 1e3)

            # solution copying and regularization parameter adjustment
            if accepted
                (plotting_callback === nothing) || plotting_callback(workset)
                swap_trajectories!(workset)
                break
            end
        end

        if !accepted || (d_∞ <= termination_threshold)
            break
        end
    end

    if logging
        return dataframe
    else
        return nothing
    end
end

