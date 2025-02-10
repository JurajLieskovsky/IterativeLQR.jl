function trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
    @unpack N = workset
    @unpack x, u, l = nominal_trajectory(workset)

    for k in 1:N
        dynamics!(x[k+1], x[k], u[k], k)
        l[k] = running_cost(x[k], u[k], k)
    end

    l[N+1] = final_cost(x[N+1], N + 1)
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

function posdef!(A)
    A .= 0.5 * (A + A')
    λ, _ = eigen(A)
    if any(λ .<= 0)
        return false
    else
        return true
    end
end

function backward_pass!(workset, μ, regularization)
    @unpack N = workset
    @unpack fx, fu = workset.dynamics_derivatives
    @unpack lx, lu, lxx, lxu, luu = workset.cost_derivatives
    @unpack Δv, vx, vxx = workset.value_function
    @unpack d, K = workset.policy_update

    for k in N:-1:1
        # value function hessian regularization
        if regularization == :state
            μI = μ * I(workset.ndx)
            vxx[k+1] += μI
        end

        # break if vxx is not positive-definite
        if !posdef!(vxx[k+1])
            return false
        end

        # perturbed argument
        qx = lx[k] + fx[k]' * vx[k+1]
        qu = lu[k] + fu[k]' * vx[k+1]

        qxx = lxx[k] + fx[k]' * vxx[k+1] * fx[k]
        quu = luu[k] + fu[k]' * vxx[k+1] * fu[k]
        qux = lxu[k]' + fu[k]' * vxx[k+1] * fx[k]

        # input hessian regularization (standard regularization as proposed by D.Q.Mayne)
        if regularization == :input
            μI = μ * I(workset.nu)
            q̃uu = quu + μI
        else
            q̃uu = quu
        end

        # break if quu is not positive-definite
        if !posdef!(q̃uu)
            return false
        end

        # controls
        d[k] = -q̃uu \ qu
        K[k] = -q̃uu \ qux

        # cost-to-go model
        vx[k] .= qx + K[k]' * qu + K[k]' * quu * d[k] + qux' * d[k]
        vxx[k] .= qxx + K[k]' * quu * K[k] + K[k]' * qux + qux' * K[k]

        # expected improvement
        Δv[k][1] = d[k]' * qu
        Δv[k][2] = 0.5 * d[k]' * quu * d[k]
    end

    return true
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

function print_iteration!(line_count, i, μ, α, J, ΔJ, Δv, accepted)
    line_count[] % 10 == 0 && @printf(
        "%-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
        "iter", "μ", "α", "∑lₖ", "∑Δlₖ", "∑Δvₖ", "accepted"
    )
    @printf(
        "%-9i %-9.3g %-9.3g %-9.3g %-9.3g %-9.3g %-9s\n",
        i, μ, α, J, ΔJ, Δv, accepted
    )
    line_count[] += 1
end

iteration_dataframe() = DataFrame(
    i=Int[], μ=Float64[], α=Float64[],
    J=Float64[], ΔJ=Float64[], ΔV=Float64[],
    accepted=Bool[]
)

function log_iteration!(dataframe, i, μ, α, J, ΔJ, Δv, accepted)
    push!(dataframe, (i, μ, α, J, ΔJ, Δv, accepted))
end
 
function regularize!(A, δ=1e-5)
    λ, v = eigen(A)
    λ .= map(e -> e < δ ? δ : e, λ)
    A .= v * diagm(λ) * v'
    return nothing
end

function iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!;
    maxiter=100, regularization=:input, ρ=0.5,
    μ=1e-4, μ_dec=3 / 4, μ_inc=4, μ_min=0.0, μ_max=1e4,
    α_values=1:-0.3:0.1, α_reg_dec=0.7, α_reg_inc=0.1,
    rollout=true, verbose=true, logging=false, plotting_callback=nothing,
    state_difference=-,
)
    # regularization parameter adjustment functions
    decrease_μ(μ) = μ * μ_dec
    increase_μ(μ) = μ * μ_inc

    # initial trajectory rollout
    rollout == true && trajectory_rollout!(workset, dynamics!, running_cost, final_cost)

    # line count for printing
    line_count = Ref(0)

    # dataframe for logging
    dataframe = logging ? iteration_dataframe() : nothing

    # algorithm
    for i in 1:maxiter
        # nominal trajectory differentiation
        differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)

        # backward pass
        if regularization == :input
            regularize!(workset.value_function.vxx[workset.N+1])
        end

        successful = backward_pass!(workset, μ, regularization)

        if !successful
            verbose && print_iteration!(line_count, i, μ, NaN, NaN, NaN, NaN, false)
            logging && log_iteration!(dataframe, i, μ, NaN, NaN, NaN, NaN, false)

            μ = increase_μ(μ)
            continue
        end

        # forward pass
        for α in α_values
            successful, J, ΔJ = forward_pass!(workset, dynamics!, state_difference, running_cost, final_cost, α)

            # error handling
            if !successful
                if α == α_values[end]
                    μ = increase_μ(μ)
                end
                continue
            end

            # iteration's evaluation
            Δv = mapreduce(Δ -> α * Δ[1] + α^2 * Δ[2], +, workset.value_function.Δv)
            accepted = ΔJ < 0 && ΔJ <= ρ * Δv

            # printout
            verbose && print_iteration!(line_count, i, μ, α, J, ΔJ, Δv, accepted)
            logging && log_iteration!(dataframe, i, μ, α, J, ΔJ, Δv, accepted)

            # solution copying and regularization parameter adjustment
            if accepted
                (plotting_callback !== nothing) && plotting_callback(workset)

                swap_trajectories!(workset)

                if α >= α_reg_dec
                    μ = decrease_μ(μ)
                    μ = μ < μ_min ? μ_min : μ
                elseif α <= α_reg_inc
                    μ = increase_μ(μ)
                end

                break
            elseif !accepted && α == α_values[end]
                μ = increase_μ(μ)
            end
        end

        # termination condition
        μ >= μ_max && break
    end

    if logging == true
        return dataframe
    else
        return nothing
    end
end

