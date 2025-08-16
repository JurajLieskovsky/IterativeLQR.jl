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

function differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!, algorithm)
    @unpack N, ndx = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack fx, fu, fxx, fux, fxu, fuu = workset.dynamics_derivatives
    @unpack lx, lu, lxx, lux, lxu, luu = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function

    @threads for k in 1:N
        if algorithm == :ilqr
            dynamics_diff!(fx[k], fu[k], x[k], u[k], k)
        elseif algorithm == :ddp
            # has not been tested
            dynamics_diff!(fx[k], fu[k], fxx[k], fxu[k], fuu[k], x[k], u[k], k)
            for i in 1:ndx
                fux[k][i,:,:] .= fxu[k][i,:,:]'
            end
        end

        running_cost_diff!(lx[k], lu[k], lxx[k], lxu[k], luu[k], x[k], u[k], k)
        lux[k] .= lxu[k]'
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1], N + 1)

    return nothing
end

function stacked_differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!, algorithm)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack ∇f, ∇2f = workset.dynamics_derivatives
    @unpack ∇l, ∇2l = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function

    @threads for k in 1:N
        if algorithm == :ilqr
            dynamics_diff!(∇f[k], x[k], u[k], k)
        elseif algorithm == :ddp
            dynamics_diff!(∇f[k], ∇2f[k], x[k], u[k], k)
        end

        running_cost_diff!(∇l[k], ∇2l[k], x[k], u[k], k)
    end

    final_cost_diff!(vx[N+1], vxx[N+1], x[N+1], N + 1)

    return nothing
end

function cost_regularization!(workset, δ)
    @unpack N = workset
    @unpack ∇2l = workset.cost_derivatives
    @unpack vxx = workset.value_function

    @threads for k in 1:N
        min_regularization!(∇2l[k], δ)
    end

    min_regularization!(vxx[N+1], δ)

    return nothing
end

function backward_pass!(workset, algorithm, δ)
    @unpack N, ndx, nu = workset
    @unpack Δv, vx, vxx = workset.value_function
    @unpack d, K = workset.policy_update
    @unpack g, qx, qu, H, qxx, quu, qux = workset.subproblem_objective_derivatives
    @unpack ∇f, ∇2f = workset.dynamics_derivatives
    @unpack ∇l, ∇2l = workset.cost_derivatives

    @inbounds for k in N:-1:1

        # gradient and hessian of the argument
        g .= ∇l[k] + ∇f[k]' * vx[k+1]
        H .= ∇2l[k] + ∇f[k]' * vxx[k+1] * ∇f[k]

        ## additional terms of the DDP algorithm
        if algorithm == :ddp
            for i in 1:ndx
                H .+= view(∇2f[k], i, :, :) * vx[k+1][i]
            end
        end

        # regularization
        isnan(δ) || min_regularization!(H, δ)

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

    Δ1 = mapreduce(Δ -> Δ[1], +, Δv)
    Δ2 = mapreduce(Δ -> Δ[2], +, Δv)

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
    i=Int[], α=Float64[], J=Float64[], ΔJ=Float64[], ΔV=Float64[], d_inf=Float64[], accepted=Bool[],
    diff=Float64[], reg=Float64[], bwd=Float64[], fwd=Float64[]
)

function log_iteration!(dataframe, i, α, J, ΔJ, Δv, d_inf, accepted, diff, reg, bwd, fwd)
    push!(dataframe, (i, α, J, ΔJ, Δv, d_inf, accepted, diff, reg, bwd, fwd))
end

function iLQR!(
    workset, dynamics!, dynamics_diff!, running_cost, running_cost_diff!, final_cost, final_cost_diff!;
    maxiter=200, ρ=1e-4, δ=sqrt(eps()), α_values=exp2.(0:-1:-16), termination_threshold=1e-4,
    rollout=true, verbose=true, logging=false, plotting_callback=nothing,
    stacked_derivatives=false, state_difference=-, regularization=:cost, algorithm=:ilqr
)
    # warn if incorrect regularization is chosen
    if algorithm == :ddp && regularization != :arg 
        @warn "`regularization=:cost` does not guarantee a succesful backward pass in the DDP algorithm. " *
        "To avoid potential errors set `regularization=:arg`."
    end
   
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
        logging && log_iteration!(dataframe, 0, NaN, J, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)

        if !successful
            return nothing
        end
    end

    # algorithm
    for i in 1:maxiter
        # nominal trajectory differentiation
        diff = @elapsed begin
            if stacked_derivatives
                stacked_differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!, algorithm)
            else
                differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!, algorithm)
            end
        end

        # regularization
        reg = (regularization == :cost) ? @elapsed(cost_regularization!(workset, δ)) : NaN

        # backward pass
        bwd = @elapsed begin
            Δv1, Δv2, d_∞ = backward_pass!(workset, algorithm, regularization == :arg ? δ : NaN)
        end

        # forward pass
        accepted = false

        for α in α_values

            fwd = @elapsed begin
                successful, J, ΔJ = forward_pass!(workset, dynamics!, state_difference, running_cost, final_cost, α)
            end

            # expected improvement and success evaluation
            Δv = α * Δv1 + α^2 * Δv2
            accepted = successful && ΔJ < 0 && ΔJ <= ρ * Δv

            # printout
            verbose && print_iteration!(line_count, i, α, J, ΔJ, Δv, d_∞, accepted, diff * 1e3, reg * 1e3, bwd * 1e3, fwd * 1e3)
            logging && log_iteration!(dataframe, i, α, J, ΔJ, Δv, d_∞, accepted, diff * 1e3, reg * 1e3, bwd * 1e3, fwd * 1e3)

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

