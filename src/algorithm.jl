function trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
    @unpack N = workset
    @unpack x, u, l = nominal_trajectory(workset)

    @inbounds for k in 1:N
        try
            dynamics!(x[k+1], x[k], u[k], k)
            l[k] = running_cost(x[k], u[k], k)
        catch
            return false, NaN
        end
    end

    l[N+1] = final_cost(x[N+1], N + 1)

    return true, sum(l)
end

function differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack fx, fu = workset.dynamics_derivatives
    @unpack lx, lu, lxx, lux, lxu, luu = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function

    @threads for k in 1:N
        dynamics_diff!(fx[k], fu[k], x[k], u[k], k)
        running_cost_diff!(lx[k], lu[k], lxx[k], lxu[k], luu[k], x[k], u[k], k)
        lux[k] .= lxu[k]'
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1], N + 1)

    return nothing
end

function stacked_differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack jac = workset.dynamics_derivatives
    @unpack grad, hess = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function

    @threads for k in 1:N
        dynamics_diff!(jac[k], x[k], u[k], k)
        running_cost_diff!(grad[k], hess[k], x[k], u[k], k)
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1], N + 1)

    return nothing
end

function cost_regularization!(workset, δ)
    @unpack N = workset
    @unpack hess = workset.cost_derivatives
    @unpack vxx = workset.value_function

    @threads for k in 1:N
        min_regularization!(hess[k], δ)
    end

    min_regularization!(vxx[N+1], δ)

    return nothing
end

function backward_pass!(workset, reg, δ)
    @unpack N, ndx, nu = workset
    @unpack Δv, vx, vxx = workset.value_function
    @unpack d, K = workset.policy_update
    @unpack g, qx, qu, H, qxx, quu, qux = workset.subproblem_objective_derivatives

    jac = workset.dynamics_derivatives.jac
    grad = workset.cost_derivatives.grad
    hess = workset.cost_derivatives.hess

    @inbounds for k in N:-1:1

        # gradient and hessian of the argument
        g .= grad[k] + jac[k]' * vx[k+1]
        H .= hess[k] + jac[k]' * vxx[k+1] * jac[k]

        # regularization
        reg && min_regularization!(H, δ)

        # control update
        F = cholesky(Symmetric(quu))
        d[k] = -(F \ qu)
        K[k] = -(F \ qux)

        # cost-to-go model
        vx[k] .= qx + K[k]' * qu
        vxx[k] .= qxx + K[k]' * qux

        # expected improvement
        Δv[k][1] = d[k]' * qu
        Δv[k][2] = 0.5 * d[k]' * quu * d[k]
    end

    Δ1 = mapreduce(Δ -> Δ[1], +, workset.value_function.Δv)
    Δ2 = mapreduce(Δ -> Δ[2], +, workset.value_function.Δv)

    d_∞ = mapreduce(d_k -> mapreduce(abs, max, d_k), max, d)

    return Δ1, Δ2, d_∞
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

        try
            dynamics!(x[k+1], x[k], u[k], k)
            l[k] = running_cost(x[k], u[k], k)
        catch
            return false, NaN, NaN
        end
    end

    l[N+1] = final_cost(x[N+1], N + 1)

    return true, sum(l), sum(l) - sum(l_ref)
end

function print_iteration!(line_count, i, α, J, ΔJ, Δv, d_inf, accepted, diff, reg, bwd, fwd)
    line_count[] % 10 == 0 && @printf(
        "%-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
        "iter", "α", "J", "ΔJ", "ΔV", "d∞", "accepted", "diff", "reg", "bwd", "fwd"
    )
    @printf(
        "%-9i %-9.3g %-9.3g %-9.3g %-9.3g %-9.3g %-9s %-9.3g %-9.3g %-9.3g %-9.3g\n",
        i, α, J, ΔJ, Δv, d_inf, accepted, diff, reg, bwd, fwd
    )
    line_count[] += 1
end

iteration_dataframe() = DataFrame(
    i=Int[], α=Float64[],
    J=Float64[], ΔJ=Float64[], ΔV=Float64[], d_inf=Float64[],
    accepted=Bool[]
)

function log_iteration!(dataframe, i, α, J, ΔJ, Δv, d_inf, accepted)
    push!(dataframe, (i, α, J, ΔJ, Δv, d_inf, accepted))
end

function iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!;
    maxiter=200, ρ=1e-4, δ=sqrt(eps()), α_values=exp2.(0:-1:-16), termination_threshold=1e-4,
    rollout=true, verbose=true, logging=false, plotting_callback=nothing,
    stacked_derivatives=false, state_difference=-, regularization=:cost
)
    # line count for printing
    line_count = Ref(0)

    # dataframe for logging
    dataframe = logging ? iteration_dataframe() : nothing

    # initial trajectory rollout
    if rollout
        rlt = @elapsed begin
            successful, J = trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
        end

        verbose && print_iteration!(line_count, 0, NaN, J, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)
        logging && log_iteration!(dataframe, 0, NaN, J, NaN, NaN, NaN, successful)

        if !successful
            return nothing
        end
    end

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

        # regularization
        reg = (regularization == :cost) ? @elapsed(cost_regularization!(workset, δ)) : NaN

        # backward pass
        bwd = @elapsed Δv1, Δv2, d_∞ = backward_pass!(workset, regularization == :arg ? true : false, δ)

        # forward pass
        accepted = false

        for α in α_values

            fwd = @elapsed begin
                successful, J, ΔJ = forward_pass!(workset, dynamics!, state_difference, running_cost, final_cost, α)
            end

            # expected improvement
            Δv = α * Δv1 + α^2 * Δv2

            # error handling
            if !successful
                verbose && print_iteration!(line_count, i, α, J, ΔJ, Δv, d_∞, false, diff * 1e3, reg * 1e3, bwd * 1e3, fwd * 1e3)
                logging && log_iteration!(dataframe, i, α, J, ΔJ, Δv, d_∞, false)
                continue
            end

            # iteration's evaluation
            accepted = ΔJ < 0 && ΔJ <= ρ * Δv

            # printout
            verbose && print_iteration!(line_count, i, α, J, ΔJ, Δv, d_∞, accepted, diff * 1e3, reg * 1e3, bwd * 1e3, fwd * 1e3)
            logging && log_iteration!(dataframe, i, α, J, ΔJ, Δv, d_∞, accepted)

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

