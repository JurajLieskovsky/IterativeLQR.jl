function trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
    @unpack N = workset
    @unpack x, u, l = nominal_trajectory(workset)

    for k in 1:N
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
    @unpack lx, lu, lxx, lxu, luu = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function

    for k in 1:N
        dynamics_diff!(fx[k], fu[k], x[k], u[k], k)
        running_cost_diff!(lx[k], lu[k], lxx[k], lxu[k], luu[k], x[k], u[k], k)
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1], N + 1)
end

function regularize(A, δ)
    λ, V = eigen(A)
    λ_reg = map(e -> e < δ ? δ : e, λ)
    return V * diagm(λ_reg) * V'
end

function backward_pass!(workset, δ)
    @unpack N, ndx, nu = workset
    @unpack fx, fu = workset.dynamics_derivatives
    @unpack lx, lu, lxx, lxu, luu = workset.cost_derivatives
    @unpack Δv, vx, vxx = workset.value_function
    @unpack d, K = workset.policy_update
    @unpack H, qxx, quu, qux, qxu = workset.subproblem_hessian

    for k in N:-1:1
        # gradient
        qx = lx[k] + fx[k]' * vx[k+1]
        qu = lu[k] + fu[k]' * vx[k+1]

        # hessian
        qxx .= lxx[k] + fx[k]' * vxx[k+1] * fx[k]
        quu .= luu[k] + fu[k]' * vxx[k+1] * fu[k]
        qux .= lxu[k]' + fu[k]' * vxx[k+1] * fx[k]
        qxu .= qux'

        # problem regularization
        tmp = copy(quu) # held for expected improvement calculation
        H .= regularize(Symmetric(H), δ)

        # control update
        F = cholesky(Symmetric(quu))
        d[k] = -(F \ qu)
        K[k] = -(F \ qux)

        # cost-to-go model
        vx[k] .= qx + K[k]' * qu
        vxx[k] .= qxx + K[k]' * qux

        # expected improvement
        Δv[k][1] = d[k]' * qu
        Δv[k][2] = 0.5 * d[k]' * tmp * d[k]
    end
end

function forward_pass!(workset, dynamics!, difference, running_cost, final_cost, α)
    @unpack N = workset
    @unpack x, u, l = active_trajectory(workset)
    @unpack d, K = workset.policy_update

    x_ref = nominal_trajectory(workset).x
    u_ref = nominal_trajectory(workset).u
    l_ref = nominal_trajectory(workset).l

    x[1] = x_ref[1]

    for k in 1:N
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

function print_iteration!(line_count, i, α, J, ΔJ, Δv, accepted)
    line_count[] % 10 == 0 && @printf(
        "%-9s %-9s %-9s %-9s %-9s %-9s\n",
        "iter", "α", "J", "ΔJ", "ΔV", "accepted"
    )
    @printf(
        "%-9i %-9.3g %-9.3g %-9.3g %-9.3g %-9s\n",
        i, α, J, ΔJ, Δv, accepted
    )
    line_count[] += 1
end

iteration_dataframe() = DataFrame(
    i=Int[], α=Float64[],
    J=Float64[], ΔJ=Float64[], ΔV=Float64[],
    accepted=Bool[]
)

function log_iteration!(dataframe, i, α, J, ΔJ, Δv, accepted)
    push!(dataframe, (i, α, J, ΔJ, Δv, accepted))
end

function iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!;
    maxiter=100, ρ=1e-4, δ=1e-5, α_values=exp.(0:-1:-15),
    rollout=true, verbose=true, logging=false, plotting_callback=nothing,
    state_difference=-,
)
    # line count for printing
    line_count = Ref(0)

    # dataframe for logging
    dataframe = logging ? iteration_dataframe() : nothing

    # initial trajectory rollout
    if rollout == true
        successful, J = trajectory_rollout!(workset, dynamics!, running_cost, final_cost)

        verbose && print_iteration!(line_count, 0, NaN, J, NaN, NaN, successful)
        logging && log_iteration!(dataframe, 0, NaN, J, NaN, NaN, successful)

        if successful == false
            return nothing
        end
    end

    # algorithm
    for i in 1:maxiter
        # nominal trajectory differentiation
        differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)

        # backward pass
        Δv = backward_pass!(workset, δ)

        # forward pass
        accepted = false

        for α in α_values
            successful, J, ΔJ = forward_pass!(workset, dynamics!, state_difference, running_cost, final_cost, α)

            # expected improvement
            Δv = mapreduce(Δ -> α * Δ[1] + α^2 * Δ[2], +, workset.value_function.Δv)

            # error handling
            if !successful
                verbose && print_iteration!(line_count, i, α, J, ΔJ, Δv, false)
                logging && log_iteration!(dataframe, i, α, J, ΔJ, Δv, false)
                continue
            end

            # iteration's evaluation
            accepted = ΔJ < 0 && ΔJ <= ρ * Δv

            # printout
            verbose && print_iteration!(line_count, i, α, J, ΔJ, Δv, accepted)
            logging && log_iteration!(dataframe, i, α, J, ΔJ, Δv, accepted)

            # solution copying and regularization parameter adjustment
            if accepted
                (plotting_callback === nothing) || plotting_callback(workset)
                swap_trajectories!(workset)
                break
            end
        end

        if accepted == false
            break
        end
    end

    if logging == true
        return dataframe
    else
        return nothing
    end
end

