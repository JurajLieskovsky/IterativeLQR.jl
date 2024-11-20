function trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
    @unpack N = workset
    @unpack x, u, l = nominal_trajectory(workset)

    for k in 1:N
        dynamics!(x[k+1], x[k], u[k])
        l[k] = running_cost(x[k], u[k])
    end

    l[N+1] = final_cost(x[N+1])
end

function differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack fx, fu = workset.dynamics_derivatives
    @unpack lx, lu, lxx, lxu, luu = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function

    for k in 1:N
        dynamics_diff!(fx[k], fu[k], x[k], u[k])
        running_cost_diff!(lx[k], lu[k], lxx[k], lxu[k], luu[k], x[k], u[k])
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1])
end

function backward_pass!(workset, μ, regularization)
    @unpack N = workset
    @unpack fx, fu = workset.dynamics_derivatives
    @unpack lx, lu, lxx, lxu, luu = workset.cost_derivatives
    @unpack Δv, vx, vxx = workset.value_function
    @unpack Δu, Δux = workset.policy_update

    if regularization == :state
        μI = μ * I(workset.nx)
    elseif regularization == :input
        μI = μ * I(workset.nu)
    end

    for k in N:-1:1
        # perturbed argument
        qx = lx[k] + fx[k]' * vx[k+1]
        qu = lu[k] + fu[k]' * vx[k+1]

        qxx = lxx[k] + fx[k]' * vxx[k+1] * fx[k]
        quu = luu[k] + fu[k]' * vxx[k+1] * fu[k]
        qux = lxu[k]' + fu[k]' * vxx[k+1] * fx[k]

        # controls
        if regularization == :input
            # standard regularization as proposed by D.Q.Mayne
            q̃uu = quu + μI
            q̃ux = qux
        elseif regularization == :state
            # regularization equivalent to vxx[k+1] += μI proposed by Y.Tassa
            q̃uu = quu + fu[k]' * μI * fu[k]
            q̃ux = qux + fu[k]' * μI * fx[k]
        end

        Δu[k] .= -q̃uu \ qu
        Δux[k] .= -q̃uu \ q̃ux

        # cost-to-go model
        vx[k] .= qx + Δux[k]' * qu + Δux[k]' * quu * Δu[k] + qux' * Δu[k]
        vxx[k] .= qxx + Δux[k]' * quu * Δux[k] + Δux[k]' * qux + qux' * Δux[k]

        # expected improvement
        Δv[k] = Δu[k]' * qu + Δu[k]' * quu * Δu[k]
    end

    return sum(Δv) # expected improvement
end

function forward_pass!(workset, dynamics!, difference, running_cost, final_cost)
    @unpack N = workset
    @unpack x, u, l = active_trajectory(workset)
    @unpack Δu, Δux = workset.policy_update

    x_ref = nominal_trajectory(workset).x
    u_ref = nominal_trajectory(workset).u
    l_ref = nominal_trajectory(workset).l

    x[1] = x_ref[1]

    for k in 1:N
        u[k] .= u_ref[k] + Δu[k] + Δux[k] * difference(x[k], x_ref[k])

        try
            dynamics!(x[k+1], x[k], u[k])
            l[k] = running_cost(x[k], u[k])
        catch
            return false, NaN, NaN
        end
    end

    l[N+1] = final_cost(x[N+1])

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

function iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!;
    maxiter=100, regularization=:input, ρ=0.5,
    μ=1e-2, μ_dec=3 / 4, μ_inc=4, μ_min=0.0, μ_max=1e4,
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
        Δv = backward_pass!(workset, μ, regularization)

        # forward pass
        for α in α_values
            successful, J, ΔJ = forward_pass!(workset, dynamics!, state_difference, running_cost, final_cost)

            # error handling
            if !successful
                μ = increase_μ(μ)
                continue
            end

            # iteration's evaluation
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

