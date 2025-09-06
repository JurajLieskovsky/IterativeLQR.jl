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
    @unpack Φx, Φxx = workset.cost_derivatives

    @threads for k in 1:N
        if algorithm == :ilqr
            dynamics_diff!(fx[k], fu[k], x[k], u[k], k)
        elseif algorithm == :ddp
            # has not been tested
            dynamics_diff!(fx[k], fu[k], fxx[k], fxu[k], fuu[k], x[k], u[k], k)
            for i in 1:ndx
                fux[k][i, :, :] .= fxu[k][i, :, :]'
            end
        end

        running_cost_diff!(lx[k], lu[k], lxx[k], lxu[k], luu[k], x[k], u[k], k)
        lux[k] .= lxu[k]'
    end

    final_cost_diff!(Φx, Φxx, x[N+1], N + 1)

    return nothing
end

function stacked_differentiation!(workset, dynamics_diff!, running_cost_diff!, final_cost_diff!, algorithm)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack ∇f, ∇2f = workset.dynamics_derivatives
    @unpack ∇l, ∇2l = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function
    @unpack Φx, Φxx = workset.cost_derivatives

    @threads for k in 1:N
        if algorithm == :ilqr
            dynamics_diff!(∇f[k], x[k], u[k], k)
        elseif algorithm == :ddp
            dynamics_diff!(∇f[k], ∇2f[k], x[k], u[k], k)
        end

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

function cost_derivatives_coordinate_transformation(workset)
    @unpack N = workset
    @unpack E, aug_E = workset.coordinate_jacobians

    full = workset.cost_derivatives
    tangent = workset.tangent_cost_derivatives

    @threads for k in 1:N
        tangent.∇l[k] .= aug_E[k]' * full.∇l[k]
        tangent.∇2l[k] .= aug_E[k]' * full.∇2l[k] * aug_E[k]
    end

    tangent.Φx .= E[N+1]' * full.Φx
    tangent.Φxx .= E[N+1]' * full.Φxx * E[N+1]

    return nothing
end

function cost_regularization!(workset, δ, regularization_approach)
    @unpack N, nx, ndx = workset
    @unpack ∇2l, Φxx = ndx == nx ? workset.cost_derivatives : workset.tangent_cost_derivatives

    @threads for k in 1:N
        regularize!(∇2l[k], δ, regularization_approach)
    end

    regularize!(Φxx, δ, regularization_approach)

    return nothing
end

function backward_pass!(workset, algorithm, regularization, δ, regularization_approach)
    @unpack N, nx, ndx, nu = workset
    @unpack Δv, vx, vxx = workset.value_function
    @unpack d, K = workset.policy_update
    @unpack g, qx, qu, H, qxx, quu, qux = workset.subproblem_objective_derivatives
    @unpack ∇f, ∇2f = workset.dynamics_derivatives
    @unpack aug_E, E = workset.coordinate_jacobians

    # cost derivatives pre-converted into the tangent space if ndx != nx
    @unpack ∇l, ∇2l = ndx == nx ? workset.cost_derivatives : workset.tangent_cost_derivatives
    @unpack Φx, Φxx = ndx == nx ? workset.cost_derivatives : workset.tangent_cost_derivatives

    vx[N+1] .= Φx
    vxx[N+1] .= Φxx

    @inbounds for k in N:-1:1

        # gradient and hessian of the argument
        if ndx == nx
            g .= ∇l[k] + ∇f[k]' * vx[k+1]
            H .= ∇2l[k] + ∇f[k]' * vxx[k+1] * ∇f[k]
        else
            g .= ∇l[k] + aug_E[k]' * (∇f[k]' * E[k+1] * vx[k+1])
            H .= ∇2l[k] + aug_E[k]' * (∇f[k]' * E[k+1] * vxx[k+1] * E[k+1]' * ∇f[k]) * aug_E[k]
        end

        ## additional tensor-vector multiplication terms of the DDP algorithm
        if algorithm == :ddp
            tensor_product = mapreduce(
                (mat, el) -> mat * el, +, eachslice(∇2f[k], dims=1), ndx == nx ? vx[k+1] : E[k+1] * vx[k+1]
            )
            tmp = ndx == nx ? tensor_product : aug_E[k]' * tensor_product * aug_E[k]

            (:ddp in regularization) && regularize!(tmp, 0, regularization_approach)
            H .+= tmp
        end

        # regularization of the entire sub-problem's Hessian
        (:arg in regularization) && regularize!(H, δ, regularization_approach)

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
    d_2 = sqrt(mapreduce(d_k -> d_k'd_k, +, d))

    return Δ1, Δ2, d_∞, d_2
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
    stacked_derivatives=false, state_difference=-, coordinate_jacobian=nothing,
    algorithm=:ilqr, regularization=(:cost, :ddp), regularization_approach=:eig
)
    @assert workset.ndx == workset.nx || coordinate_jacobian !== nothing

    # line count for printing
    line_count = Ref(0)

    # dataframe for logging
    dataframe = logging ? iteration_dataframe() : nothing

    # initial trajectory rollout
    if rollout == :full
        rlt = @elapsed begin
            successful, J = trajectory_rollout!(workset, dynamics!, running_cost, final_cost)
        end

        verbose && print_iteration!(line_count, 0, NaN, J, NaN, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)
        logging && log_iteration!(dataframe, 0, NaN, J, NaN, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)

        if !successful
            return nothing
        end
    elseif rollout == :partial
        for α in α_values
            rlt = @elapsed begin
                successful, J, ΔJ = forward_pass!(workset, dynamics!, state_difference, running_cost, final_cost, α)
            end

            verbose && print_iteration!(line_count, 0, NaN, J, ΔJ, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)
            logging && log_iteration!(dataframe, 0, NaN, J, ΔJ, NaN, NaN, NaN, successful, NaN, NaN, NaN, rlt * 1e3)

            if successful
                swap_trajectories!(workset)
                break
            end
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

        # coordinate jacobian calculation
        coordinate_jacobian !== nothing && coordinate_jacobian_calculation(workset, coordinate_jacobian)

        # conversion of cost derivatives into tangential plane
        workset.ndx != workset.nx && cost_derivatives_coordinate_transformation(workset)

        # regularization
        reg = :cost in regularization ? @elapsed(cost_regularization!(workset, δ, regularization_approach)) : NaN

        # backward pass
        bwd = @elapsed begin
            Δv1, Δv2, d_∞, d_2 = backward_pass!(workset, algorithm, regularization, δ, regularization_approach)
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

