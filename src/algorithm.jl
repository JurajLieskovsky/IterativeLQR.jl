function trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
    @unpack N = workset
    @unpack x, u, l, p = nominal_trajectory(workset)
    @unpack terminal_state_projection, terminal_state_constraint = workset.constraints
    @unpack input_projection, input_constraint = workset.constraints
    @unpack state_projection, state_constraint = workset.constraints

    @inbounds for k in 1:N
        try
            dynamics!(x[k+1], x[k], u[k], k)
            l[k] = running_cost(x[k], u[k], k)

            if !isnothing(input_projection)
                p[k] = evaluate_penalty(input_constraint[k], u[k])
            end

            if !isnothing(state_projection)
                p[k] += evaluate_penalty(state_constraint[k], x[k])
            end
        catch
            return false
        end
    end

    l[N+1] = final_cost(x[N+1], N + 1)

    if !isnothing(terminal_state_projection)
        p[N+1] = evaluate_penalty(terminal_state_constraint, x[N+1])
    end

    return true, sum(l), sum(p)
end

function differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack fx, fu = workset.dynamics_derivatives
    @unpack lx, lu, lxx, lux, lxu, luu = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function

    @unpack terminal_state_projection, terminal_state_constraint = workset.constraints
    @unpack input_projection, input_constraint = workset.constraints
    @unpack state_projection, state_constraint = workset.constraints

    @threads for k in 1:N
        dynamics_diff!(fx[k], fu[k], x[k], u[k], k)
        running_cost_diff!(lx[k], lu[k], lxx[k], lxu[k], luu[k], x[k], u[k], k)
        isnothing(input_projection) || add_penalty_derivative!(lu[k], luu[k], input_constraint[k], u[k])
        isnothing(state_projection) || add_penalty_derivative!(lx[k], lxx[k], state_constraint[k], x[k])
        lux[k] .= lxu[k]'
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1], N + 1)
    isnothing(terminal_state_projection) || add_penalty_derivative!(vx[N+1], vxx[N+1], terminal_state_constraint, x[N+1])

    return nothing
end

function stacked_differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack jac = workset.dynamics_derivatives
    @unpack grad, hess = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function

    @unpack terminal_state_projection, terminal_state_constraint = workset.constraints
    @unpack input_projection, input_constraint = workset.constraints
    @unpack state_projection, state_constraint = workset.constraints
    @unpack lx, lxx, lu, luu = workset.cost_derivatives

    @threads for k in 1:N
        dynamics_diff!(jac[k], x[k], u[k], k)
        running_cost_diff!(grad[k], hess[k], x[k], u[k], k)
        isnothing(input_projection) || add_penalty_derivative!(lu[k], luu[k], input_constraint[k], u[k])
        isnothing(state_projection) || add_penalty_derivative!(lx[k], lxx[k], state_constraint[k], x[k])
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1], N + 1)
    isnothing(terminal_state_projection) || add_penalty_derivative!(vx[N+1], vxx[N+1], terminal_state_constraint, x[N+1])

    return nothing
end

function regularization!(workset, regularization_function!)
    @unpack N = workset
    @unpack hess = workset.cost_derivatives
    @unpack vxx = workset.value_function

    @threads for k in 1:N
        regularization_function!(hess[k])
    end

    regularization_function!(vxx[N+1])

    return nothing
end

# backward and forward pass of the algorithm

function backward_pass!(workset)
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

        # control update
        F = cholesky(Symmetric(quu))
        d[k] = -(F \ qu)
        K[k] = -(F \ qux)

        # cost-to-go model
        vx[k] .= qx + K[k]' * qu
        vxx[k] .= qxx - K[k]' * quu * K[k]

        # expected improvement
        Δv[k][1] = d[k]' * qu
        Δv[k][2] = 0.5 * d[k]' * quu * d[k]
    end

    return nothing
end

function trajectory_update!(workset, dynamics!, difference, α)
    @unpack N = workset
    @unpack x, u = active_trajectory(workset)
    @unpack d, K = workset.policy_update

    x_ref = nominal_trajectory(workset).x
    u_ref = nominal_trajectory(workset).u

    x[1] = x_ref[1]

    @inbounds for k in 1:N
        u[k] .= u_ref[k] + α * d[k] + K[k] * difference(x[k], x_ref[k])
        try
            dynamics!(x[k+1], x[k], u[k], k)
        catch
            return false
        end
    end

    return true
end

function trajectory_evaluation!(workset, running_cost, final_cost)
    @unpack N = workset
    @unpack x, u, l, p = active_trajectory(workset)
    @unpack terminal_state_projection, terminal_state_constraint = workset.constraints
    @unpack input_projection, input_constraint = workset.constraints
    @unpack state_projection, state_constraint = workset.constraints

    @inbounds @threads for k in 1:N
        l[k] = running_cost(x[k], u[k], k)

        if !isnothing(input_projection)
            p[k] = evaluate_penalty(input_constraint[k], u[k])
        end

        if !isnothing(state_projection)
            p[k] += evaluate_penalty(state_constraint[k], x[k])
        end
    end

    l[N+1] = final_cost(x[N+1], N + 1)

    if !isnothing(terminal_state_projection)
        p[N+1] = evaluate_penalty(terminal_state_constraint, x[N+1])
    end

    return sum(l), sum(p)
end

function slack_and_dual_variable_update!(workset, adaptive)
    @unpack N = workset
    @unpack terminal_state_projection, terminal_state_constraint = workset.constraints
    @unpack input_projection, input_constraint = workset.constraints
    @unpack state_projection, state_constraint = workset.constraints
    @unpack x, u, p = nominal_trajectory(workset)

    rk_∞, sk_∞ = 0, 0

    if !isnothing(input_projection)
        @inbounds @threads for k in 1:N
            update_slack_and_dual_variable!(input_projection, input_constraint[k], u[k], adaptive)
            p[k] = evaluate_penalty(input_constraint[k], u[k])
        end
        rk_∞ = max(rk_∞, mapreduce(c -> norm(c.r, Inf), max, input_constraint))
        sk_∞ = max(sk_∞, mapreduce(c -> norm(c.s, Inf), max, input_constraint))
    end

    if !isnothing(state_projection)
        @inbounds @threads for k in 1:N
            update_slack_and_dual_variable!(state_projection, state_constraint[k], x[k], adaptive)
            p[k] += evaluate_penalty(state_constraint[k], x[k])
        end
        rk_∞ = max(rk_∞, mapreduce(c -> norm(c.r, Inf), max, state_constraint))
        sk_∞ = max(sk_∞, mapreduce(c -> norm(c.s, Inf), max, state_constraint))
    end

    rN_∞, sN_∞ = 0, 0

    if !isnothing(terminal_state_projection)
        update_slack_and_dual_variable!(terminal_state_projection, terminal_state_constraint, x[N+1], adaptive)
        p[N+1] = evaluate_penalty(terminal_state_constraint, x[N+1])

        rN_∞ = max(rN_∞, norm(terminal_state_constraint.r, Inf))
        sN_∞ = max(sN_∞, norm(terminal_state_constraint.s, Inf))
    end

    # print update
    @printf("%-9s %-9s %-9s %-9s\n", "rk_∞", "sk_∞", "rN_∞", "sN_∞")
    @printf("%-9.3g %-9.3g %-9.3g %-9.3g\n", rk_∞, sk_∞, rN_∞, sN_∞)

    return sum(p)
end

# printing and saving utilities

function print_iteration!(line_count, j, i, α, J, P, ΔJ, ΔP, Δv, l_inf, l_2, accepted, timer)
    line_count[] % 10 == 0 && @printf(
        "%-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
        "outer", "inner", "α", "J", "P", "ΔJ", "ΔP", "ΔV", "l∞", "l2", "accepted",
        "diff", "reg", "bwd", "fwd", "eval"
    )
    @printf(
        "%-9i %-9i %-9.3g %-9.3g %-9.3g %-9.3g %-9.3g %-9.3g %-9.3g %-9.3g %-9s %-9.3g %-9.3g %-9.3g %-9.3g %-9.3g\n",
        j, i, α, J, P, ΔJ, ΔP, Δv, l_inf, l_2, accepted,
        timer[:diff] * 1e3, timer[:reg] * 1e3, timer[:bwd] * 1e3, timer[:fwd] * 1e3, timer[:eval] * 1e3

    )
    line_count[] += 1
end

iteration_dataframe() = DataFrame(
    j=Int[], i=Int[], α=Float64[],
    J=Float64[], P=Float64[], ΔJ=Float64[], ΔP=Float64[], ΔV=Float64[],
    l_inf=Float64[], l_2=Float64[],
    accepted=Bool[]
)

function log_iteration!(dataframe, j, i, α, J, P, ΔJ, ΔP, Δv, l_inf, l_2, accepted)
    push!(dataframe, (j, i, α, J, P, ΔJ, ΔP, Δv, l_inf, l_2, accepted))
end

# algorithm
function iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!;
    maxouter=100, maxinner=20, σ=1e-4, δ=sqrt(eps()), α_values=exp2.(0:-1:-16), l∞_threshold=1e-6, adaptive=:mul,
    rollout=true, verbose=true, logging=false, plotting_callback=nothing,
    stacked_derivatives=false, state_difference=-, regularization=:min
)
    # line count for printing
    line_count = Ref(0)
    timer = Dict(:diff => NaN, :reg => NaN, :bwd => NaN, :fwd => NaN, :eval => NaN)

    # dataframe for logging
    dataframe = logging ? iteration_dataframe() : nothing

    # regularization function
    regularization_function! =
        if regularization == :min
            H -> min_regularization!(H, δ)
        elseif regularization == :flip
            H -> flip_regularization!(H, δ)
        elseif regularization == :holy
            holy_regularization!
        end

    # initial trajectory rollout
    if rollout
        # rollout trajectory
        timer[:fwd] = @elapsed successful, ref_J, ref_P = trajectory_rollout!(workset, dynamics!, running_cost, final_cost)

        # update constraint
        ref_P = slack_and_dual_variable_update!(workset, :none)

        # print and log
        verbose && print_iteration!(line_count, 0, 0, NaN, ref_J, ref_P, NaN, NaN, NaN, NaN, NaN, successful, timer)
        logging && log_iteration!(dataframe, 0, 0, NaN, ref_J, ref_P, NaN, NaN, NaN, NaN, NaN, successful)

        # plot trajectory
        (plotting_callback === nothing) || plotting_callback(workset)

        if !successful
            return nothing
        end
    end

    # algorithm
    for j in 1:maxouter
        for i in 1:maxinner
            # nominal trajectory differentiation
            timer[:diff] = @elapsed begin
                if stacked_derivatives
                    stacked_differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
                else
                    differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
                end
            end

            # regularization
            timer[:reg] = regularization == :none ? NaN : @elapsed regularization!(workset, regularization_function!)

            # backward pass
            timer[:bwd] = @elapsed backward_pass!(workset)

            # l_inf and l_2 norms of policy update
            l∞ = mapreduce(d -> norm(d, Inf), max, workset.policy_update.d)
            l2 = sqrt(mapreduce(d -> sum(d .^ 2), +, workset.policy_update.d))

            # forward pass
            accepted = false

            for α in α_values

                timer[:fwd] = @elapsed successful = trajectory_update!(workset, dynamics!, state_difference, α)

                if !successful
                    Δv, J, P, ΔJ, ΔP, accepted, eval = NaN, NaN, NaN, NaN, NaN, false, NaN
                else
                    # expected improvement
                    Δv = mapreduce(Δ -> α * Δ[1] + α^2 * Δ[2], +, workset.value_function.Δv)

                    # cost evaluation
                    timer[:eval] = @elapsed J, P = trajectory_evaluation!(workset, running_cost, final_cost)

                    # total cost and penalty sum
                    ΔJ = J - ref_J 
                    ΔP = P - ref_P

                    # iteration's evaluation
                    accepted = (ΔJ + ΔP) < 0 && (ΔJ + ΔP) <= σ * Δv
                end

                # print and log
                verbose && print_iteration!(line_count, j, i, α, J, P, ΔJ, ΔP, Δv, l∞, l2, accepted, timer)
                logging && log_iteration!(dataframe, j, i, α, J, P, ΔJ, ΔP, Δv, l∞, l2, accepted)

                # swap trajectories and plot
                if accepted
                    ref_J, ref_P = J, P
                    swap_trajectories!(workset) # swap nominal trajectory for active trajectory
                    (plotting_callback === nothing) || plotting_callback(workset) # plot trajectory
                    break
                end
            end

            if (l∞ <= l∞_threshold) || !accepted
                break
            end
        end

        # update slack and dual variable (based on nominal trajectory)
        ref_P = slack_and_dual_variable_update!(workset, adaptive)
    end

    if logging
        return dataframe
    else
        return nothing
    end
end

