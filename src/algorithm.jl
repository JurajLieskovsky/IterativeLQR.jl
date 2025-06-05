function trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
    @unpack N = workset
    @unpack x, u, l = nominal_trajectory(workset)

    @inbounds for k in 1:N
        try
            dynamics!(x[k+1], x[k], u[k], k)
            l[k] = running_cost(x[k], u[k], k)
        catch
            return false
        end
    end

    l[N+1] = final_cost(x[N+1], N + 1)

    return true
end

function differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack fx, fu = workset.dynamics_derivatives
    @unpack lx, lu, lxx, lux, lxu, luu = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function

    @unpack terminal_state_projection, terminal_state_constraint = workset.constraints
    @unpack input_projection, input_constraint = workset.constraints

    @threads for k in 1:N
        dynamics_diff!(fx[k], fu[k], x[k], u[k], k)
        running_cost_diff!(lx[k], lu[k], lxx[k], lxu[k], luu[k], x[k], u[k], k)
        isnothing(input_projection) || add_penalty_derivative!(lu[k], luu[k], u[k], input_constraint[k])
        lux[k] .= lxu[k]'
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1], N + 1)
    isnothing(terminal_state_projection) || add_penalty_derivative!(vx[N+1], vxx[N+1], x[N+1], terminal_state_constraint)

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
    @unpack lu, luu = workset.cost_derivatives

    @threads for k in 1:N
        dynamics_diff!(jac[k], x[k], u[k], k)
        running_cost_diff!(grad[k], hess[k], x[k], u[k], k)
        isnothing(input_projection) || add_penalty_derivative!(lu[k], luu[k], u[k], input_constraint[k])
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1], N + 1)
    isnothing(terminal_state_projection) || add_penalty_derivative!(vx[N+1], vxx[N+1], x[N+1], terminal_state_constraint)

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

        # hold quu for expected improvement calculation
        tmp = copy(quu)

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

    x_ref = nominal_trajectory(workset).x
    u_ref = nominal_trajectory(workset).u
    p_ref = nominal_trajectory(workset).p

    @inbounds @threads for k in 1:N
        l[k] = running_cost(x[k], u[k], k)

        if !isnothing(input_projection)
            p[k] = evaluate_penalty(u[k], input_constraint[k])

            if isdirty(nominal_trajectory(workset))
                p_ref[k] = evaluate_penalty(u_ref[k], input_constraint[k])
            end
        end
    end

    l[N+1] = final_cost(x[N+1], N + 1)

    if !isnothing(terminal_state_projection)
        p[N+1] = evaluate_penalty(x[N+1], terminal_state_constraint)

        if isdirty(nominal_trajectory(workset))
            p_ref[N+1] = evaluate_penalty(x_ref[N+1], terminal_state_constraint)
        end
    end

    return nothing
end

function slack_and_dual_variable_update!(workset)
    @unpack N = workset
    @unpack terminal_state_projection, terminal_state_constraint = workset.constraints
    @unpack input_projection, input_constraint = workset.constraints
    @unpack x, u = nominal_trajectory(workset)

    if !isnothing(terminal_state_projection)
        update_slack_and_dual_variable!(terminal_state_projection, x[N+1], terminal_state_constraint)
    end

    if !isnothing(input_projection)
        @inbounds @threads for k in 1:N
            update_slack_and_dual_variable!(input_projection, u[k], input_constraint[k])
        end
    end

    for trajectory in workset.trajectory
        trajectory.isdirty[] = true
    end

    return nothing
end

# printing and saving utilities

function print_iteration!(line_count, i, α, J, P, ΔJ, ΔP, Δv, accepted, diff, reg, bwd, fwd)
    line_count[] % 10 == 0 && @printf(
        "%-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
        "iter", "α", "J", "P", "ΔJ", "ΔP", "ΔV", "accepted", "diff", "reg", "bwd", "fwd"
    )
    @printf(
        "%-9i %-9.3g %-9.3g %-9.3g %-9.3g %-9.3g %-9.3g %-9s %-9.3g %-9.3g %-9.3g %-9.3g\n",
        i, α, J, P, ΔJ, ΔP, Δv, accepted, diff, reg, bwd, fwd
    )
    line_count[] += 1
end

iteration_dataframe() = DataFrame(
    i=Int[], α=Float64[],
    J=Float64[], P=Float64[], ΔJ=Float64[], ΔP=Float64[], ΔV=Float64[],
    accepted=Bool[]
)

function log_iteration!(dataframe, i, α, J, P, ΔJ, ΔP, Δv, accepted)
    push!(dataframe, (i, α, J, P, ΔJ, ΔP, Δv, accepted))
end

# algorithm
function iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!;
    maxiter=100, σ=1e-4, δ=sqrt(eps()), α_values=exp2.(0:-1:-16),
    rollout=true, verbose=true, logging=false, plotting_callback=nothing,
    stacked_derivatives=false, state_difference=-, regularization=:min
)
    # line count for printing
    line_count = Ref(0)

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
        rlt = @elapsed begin
            successful = trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
        end

        # calculate preliminary total cost
        J = sum(nominal_trajectory(workset).l)

        # update slack and dual variables
        slack_and_dual_variable_update!(workset)

        # print and log
        verbose && print_iteration!(line_count, 0, NaN, J, NaN, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)
        logging && log_iteration!(dataframe, 0, NaN, J, NaN, NaN, NaN, NaN, successful)

        # plot trajectory
        (plotting_callback === nothing) || plotting_callback(workset)

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
        reg = (regularization == :none) ? NaN : @elapsed regularization!(workset, regularization_function!)

        # backward pass
        bwd = @elapsed backward_pass!(workset)

        # forward pass
        accepted = false

        for α in α_values

            fwd = @elapsed begin
                successful = trajectory_update!(workset, dynamics!, state_difference, α)
                trajectory_evaluation!(workset, running_cost, final_cost)
            end

            # total cost and penalty sum
            J = sum(active_trajectory(workset).l)
            P = sum(active_trajectory(workset).p)

            ΔJ = J - sum(nominal_trajectory(workset).l)
            ΔP = P - sum(nominal_trajectory(workset).p)

            # expected improvement
            Δv = mapreduce(Δ -> α * Δ[1] + α^2 * Δ[2], +, workset.value_function.Δv)

            # error handling
            if !successful
                verbose && print_iteration!(line_count, i, α, J, P, ΔJ, ΔP, Δv, false, diff * 1e3, reg * 1e3, bwd * 1e3, fwd * 1e3)
                logging && log_iteration!(dataframe, i, α, J, P, ΔJ, ΔP, Δv, false)
                continue
            end

            # iteration's evaluation
            accepted = (ΔJ + ΔP) < 0 && (ΔJ + ΔP) <= σ * Δv

            # print and log
            verbose && print_iteration!(line_count, i, α, J, P, ΔJ, ΔP, Δv, accepted, diff * 1e3, reg * 1e3, bwd * 1e3, fwd * 1e3)
            logging && log_iteration!(dataframe, i, α, J, P, ΔJ, ΔP, Δv, accepted)

            # solution copying and regularization parameter adjustment
            if accepted
                # plot trajectory
                (plotting_callback === nothing) || plotting_callback(workset)

                # swap nominal trajectory for active trajectory
                swap_trajectories!(workset)

                # update slack and dual variable (based on nominal trajectory)
                slack_and_dual_variable_update!(workset)

                break
            end
        end

        if !accepted
            break
        end
    end

    if logging
        return dataframe
    else
        return nothing
    end
end

