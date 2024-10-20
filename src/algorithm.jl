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
    @unpack vx, vxx = workset.cost_to_go

    for k in 1:N
        dynamics_diff!(fx[k], fu[k], x[k], u[k])
        running_cost_diff!(lx[k], lu[k], lxx[k], lxu[k], luu[k], x[k], u[k])
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1])
end

function backward_pass!(workset, μ)
    @unpack N = workset
    @unpack fx, fu = workset.dynamics_derivatives
    @unpack lx, lu, lxx, lxu, luu = workset.cost_derivatives
    @unpack Δv, vx, vxx = workset.cost_to_go
    @unpack Δu, Δux = workset.policy_update

    μI = μ * I(workset.nx)

    for k in N:-1:1
        # perturbed argument
        qx = lx[k] + fx[k]' * vx[k+1]
        qu = lu[k] + fu[k]' * vx[k+1]

        qxx = lxx[k] + fx[k]' * vxx[k+1] * fx[k]
        quu = luu[k] + fu[k]' * vxx[k+1] * fu[k]
        qux = lxu[k]' + fu[k]' * vxx[k+1] * fx[k]

        # controls (with regularization equivalent to vxx[k+1] .+= μI)
        q̃uu = quu + fu[k]' * μI * fu[k]
        q̃ux = qux + fu[k]' * μI * fx[k]

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

function forward_pass!(workset, dynamics!, running_cost, final_cost)
    @unpack N = workset
    @unpack x, u, l = active_trajectory(workset)
    @unpack Δu, Δux = workset.policy_update

    x_ref = nominal_trajectory(workset).x
    u_ref = nominal_trajectory(workset).u
    l_ref = nominal_trajectory(workset).l

    x[1] = x_ref[1]

    for k in 1:N
        u[k] .= u_ref[k] + Δu[k] + Δux[k] * (x[k] - x_ref[k])

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

# main algorithm

function iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!;
    maxiter=200, μ=10.0^0, ρ=0.5, ψs=0.70, ψf=5,
    rollout=true, verbose=true, plotting_callback=nothing
)
    # initial trajectory rollout
    rollout == true && trajectory_rollout!(workset, dynamics!, running_cost, final_cost)

    # algorithm
    for i = 1:maxiter
        # header 
        verbose && (i - 1) % 5 == 0 && @printf(
                "%-9s %-9s %-9s %-9s %-9s %-9s\n",
                "iter", "μ", "∑lₖ", "∑Δlₖ", "∑Δvₖ", "accepted"
            )

        # nominal trajectory differentiation
        differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!)

        # backward pass
        expected_improvement = backward_pass!(workset, μ)

        # forward pass
        successful, cost, improvement = forward_pass!(workset, dynamics!, running_cost, final_cost)

        # pre-decision printout
        verbose && @printf(
            "%-9i %-9.3g %-9.3g %-9.3g %-9.3g ",
            i, μ, cost, improvement, expected_improvement
        )

        # accept/reject new trajectory
        if (improvement <= ρ * expected_improvement) && successful
            swap_trajectories!(workset)

            μ *= ψs
            μ = (μ > 0) ? μ : 0

            verbose && @printf("%-9s\n", "true")

            (plotting_callback !== nothing) && plotting_callback(workset) 
        else
            μ *= ψf

            verbose && @printf("%-9s\n", "false")
        end
    end
end

