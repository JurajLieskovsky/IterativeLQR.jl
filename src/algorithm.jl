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

    return true
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

function forward_pass!(workset, dynamics!, difference, running_cost, final_cost, α)
    @unpack N = workset
    @unpack x, u, l = active_trajectory(workset)
    @unpack d, K = workset.policy_update

    x_ref = nominal_trajectory(workset).x
    u_ref = nominal_trajectory(workset).u

    x[1] = x_ref[1]

    @inbounds for k in 1:N
        u[k] .= u_ref[k] + α * d[k] + K[k] * difference(x[k], x_ref[k])

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

# constraint related functions
 
function evaluate_penalties!(workset)
    @unpack N = workset
    @unpack terminal_state_constraint, input_constraint = workset

    for trajectory in workset.trajectory
        # continue if slack and dual variables are unchanged
        trajectory.dirty[] || continue

        # fill per-step penalties with zeros
        fill!(trajectory.p, 0)

        # terminal state constraint
        if terminal_state_constraint.projection !== nothing
            trajectory.p[N+1] = evaluate_penalty(terminal_state_constraint, trajectory.x[N+1])
        end

        # input constraint
        if input_constraint.projection !== nothing
            @views evaluate_penalty!(trajectory.p[1:N], input_constraint, trajectory.u)
        end
    end

    return nothing
end

function add_penalty_derivatives!(workset)
    @unpack N = workset
    @unpack x,u = nominal_trajectory(workset)
    @unpack vx, vxx = workset.value_function
    @unpack lu, luu = workset.cost_derivatives
    @unpack terminal_state_constraint, input_constraint = workset

    if terminal_state_constraint.projection !== nothing
        add_penalty_derivative!(vx[N+1], vxx[N+1], terminal_state_constraint, x[N+1])
    end

    if input_constraint.projection !== nothing
        add_penalty_derivative!(lu, luu, input_constraint, u)
    end

    return nothing
end

function update_slack_and_dual_variables!(workset)
    @unpack N = workset
    @unpack x,u = nominal_trajectory(workset)
    @unpack terminal_state_constraint, input_constraint = workset

    if terminal_state_constraint.projection !== nothing
        update_slack_and_dual_variable!(terminal_state_constraint, x[N+1])
    end

    if input_constraint.projection !== nothing
        update_slack_and_dual_variable!(input_constraint, u)
    end

    for trajectory in workset.trajectory
        trajectory.dirty[] = true
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
        update_slack_and_dual_variables!(workset)

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

        # add terminal constaint penalty's derivatives
        add_penalty_derivatives!(workset)

        # regularization
        reg = (regularization == :none) ? NaN : @elapsed regularization!(workset, regularization_function!)

        # backward pass
        bwd = @elapsed backward_pass!(workset)

        # forward pass
        accepted = false

        for α in α_values

            fwd = @elapsed begin
                successful = forward_pass!(workset, dynamics!, state_difference, running_cost, final_cost, α)
            end

            # add terminal constraint penalty
            evaluate_penalties!(workset)

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
                update_slack_and_dual_variables!(workset)

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

