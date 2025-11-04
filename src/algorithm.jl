function trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
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

function differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
    @unpack N, ndx = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack fx, fu, fxx, fux, fxu, fuu = workset.dynamics_derivatives
    @unpack lx, lu, lxx, lux, lxu, luu = workset.cost_derivatives
    @unpack Φx, Φxx = workset.cost_derivatives

    @threads for k in 1:N
        dynamics_diff!(fx[k], fu[k], x[k], u[k], k)
        running_cost_diff!(lx[k], lu[k], lxx[k], lxu[k], luu[k], x[k], u[k], k)
        lux[k] .= lxu[k]'
    end

    final_cost_diff!(Φx, Φxx, x[N+1], N + 1)

    return nothing
end

function stacked_differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack ∇f = workset.dynamics_derivatives
    @unpack ∇l, ∇2l = workset.cost_derivatives
    @unpack Φx, Φxx = workset.cost_derivatives

    @threads for k in 1:N
        dynamics_diff!(∇f[k], x[k], u[k], k)
        running_cost_diff!(∇l[k], ∇2l[k], x[k], u[k], k)
    end

    final_cost_diff!(Φx, Φxx, x[N+1], N + 1)

    return nothing
end

function coordinate_jacobian_calculation(workset, coordinate_jacobian)
    @unpack N = workset
    @unpack x = nominal_trajectory(workset)
    @unpack E = workset.coordinate_jacobians

    @threads for k in 1:N+1
        E[k] .= coordinate_jacobian(x[k])
    end
end

function derivatives_coordinate_transformation(workset)
    @unpack N = workset
    @unpack E = workset.coordinate_jacobians

    dyn = workset.dynamics_derivatives
    cost = workset.cost_derivatives
    tan_dyn = workset.tangent_dynamics_derivatives
    tan_cost = workset.tangent_cost_derivatives

    @threads for k in 1:N
        # dynamics
        tan_dyn.fx[k] .= E[k+1]' * dyn.fx[k] * E[k]
        tan_dyn.fu[k] .= E[k+1]' * dyn.fu[k]

        # running cost
        tan_cost.lx[k] .= E[k]' * cost.lx[k]
        tan_cost.lu[k] .= cost.lu[k]

        tan_cost.lxx[k] .= E[k]' * cost.lxx[k] * E[k]
        tan_cost.lxu[k] .= E[k]' * cost.lxu[k]
        tan_cost.lux[k] .= cost.lux[k] * E[k]
        tan_cost.luu[k] .= cost.luu[k]
    end

    # final cost
    tan_cost.Φx .= E[N+1]' * cost.Φx
    tan_cost.Φxx .= E[N+1]' * cost.Φxx * E[N+1]

    return nothing
end

function cost_regularization!(workset, δ, regularization)
    @unpack N, nx, ndx = workset
    @unpack ∇2l, Φxx = ndx == nx ? workset.cost_derivatives : workset.tangent_cost_derivatives

    @threads for k in 1:N
        regularize!(∇2l[k], δ, regularization)
    end

    regularize!(Φxx, δ, regularization)

    return nothing
end

function backward_pass!(workset)
    @unpack N, nx, ndx, nu = workset
    @unpack d, K = workset.policy_update
    @unpack vx, vxx = workset.backward_pass_workset
    @unpack g, qx, qu, H, qxx, quu, qux = workset.backward_pass_workset

    # cost derivatives pre-converted into the tangent space if ndx != nx
    @unpack ∇f = ndx == nx ? workset.dynamics_derivatives : workset.tangent_dynamics_derivatives
    @unpack ∇l, ∇2l = ndx == nx ? workset.cost_derivatives : workset.tangent_cost_derivatives
    @unpack Φx, Φxx = ndx == nx ? workset.cost_derivatives : workset.tangent_cost_derivatives

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

function forward_pass!(workset, dynamics!, difference, running_cost, final_cost, α)
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

iteration_dataframe() = DataFrame(
    i=Int[], α=Float64[], J=Float64[], ΔJ=Float64[], ΔV=Float64[], d_inf=Float64[], d_2=Float64[], accepted=Bool[],
    diff=Float64[], reg=Float64[], bwd=Float64[], fwd=Float64[]
)

function log_iteration!(dataframe, i, α, J, ΔJ, Δv, d_inf, d_2, accepted, diff, reg, bwd, fwd)
    push!(dataframe, (i, α, J, ΔJ, Δv, d_inf, d_2, accepted, diff, reg, bwd, fwd))
end

function iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!;
    maxiter=250, ρ=1e-4, δ=sqrt(eps()), α_values=exp2.(0:-1:-16), termination_threshold=1e-4,
    rollout=:full, verbose=true, logging=false, plotting_callback=nothing,
    stacked_derivatives=false, state_difference=-, coordinate_jacobian=nothing, regularization=:mchol
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
            if stacked_derivatives
                stacked_differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
            else
                differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
            end
        end

        # coordinate jacobian calculation
        coordinate_jacobian !== nothing && coordinate_jacobian_calculation(workset, coordinate_jacobian)

        # conversion of cost derivatives into tangential plane
        if workset.ndx != workset.nx
            derivatives_coordinate_transformation(workset)
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

