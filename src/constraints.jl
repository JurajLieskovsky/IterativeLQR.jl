# Terminal state constraint
mutable struct Constraints{T}
    ρ::T

    z_projection::Union{Function,Nothing}
    z::Vector{T}
    α::Vector{T}

    w_projection::Union{Function,Nothing}
    w::Vector{Vector{T}}
    β::Vector{Vector{T}}

    function Constraints{T}(nx, nu, N) where {T}
        z = zeros(T, nx)
        α = zeros(T, nx)
        β = [zeros(T, nu) for _ in 1:N]
        w = [zeros(T, nu) for _ in 1:N]

        new(zero(T), nothing, α, z, nothing, β, w)
    end
end

## set functions

function set_terminal_constraint_projection_function!(workset, fun)
    workset.constraints.z_projection = fun
end

function set_input_constraint_projection_function!(workset, fun)
    workset.constraints.w_projection = fun
end

function set_penalty_parameter!(workset, ρ_new)
    @unpack α, β = workset.constraints

    ratio = workset.constraints.ρ / ρ_new
    workset.constraints.ρ = ρ_new

    α .*= ratio

    for λ in β
        λ .*= ratio
    end

    return nothing
end

## general evaluation functions
evaluate_penalty(ρ, arg) = ρ / 2 * mapreduce(a -> a^2, +, arg)

function add_penalty_derivative!(grad, hess, ρ, arg)
    grad .+= ρ * arg
    hess[diagind(hess)] .+= ρ
end

# workset evaluation functions
function add_penalty_derivatives!(workset)
    @unpack N = workset
    @unpack x, u = nominal_trajectory(workset)
    @unpack lu, luu = workset.cost_derivatives
    @unpack vx, vxx = workset.value_function

    @unpack ρ = workset.constraints
    @unpack z_projection, z, α = workset.constraints
    @unpack w_projection, w, β = workset.constraints

    if z_projection !== nothing
        add_penalty_derivative!(vx[N+1], vxx[N+1], ρ, x[N+1] - z + α)
    end

    if w_projection !== nothing
        for k in 1:N
            add_penalty_derivative!(lu[k], luu[k], ρ, u[k] - w[k] + β[k])
        end
    end

    return nothing
end

function evaluate_penalties(workset, trajectory)
    @unpack N = workset
    @unpack ρ = workset.constraints
    @unpack z_projection, z, α = workset.constraints
    @unpack w_projection, w, β = workset.constraints

    penalty = (z_projection !== nothing) ? evaluate_penalty(ρ, trajectory.x[N+1] - z + α) : 0

    if (w_projection !== nothing)
        for k in 1:N
            penalty += evaluate_penalty(ρ, trajectory.u[k] - w[k] + β[k])
        end
    end

    return penalty
end

function update_slack_variables!(workset, trajectory)
    @unpack N = workset
    @unpack z_projection, z, α = workset.constraints
    @unpack w_projection, w, β = workset.constraints

    if z_projection !== nothing
        z .= z_projection(trajectory.x[N+1] + α)
    end

    if w_projection !== nothing
        for k in 1:N
            w[k] .= w_projection(trajectory.u[k] + β[k])
        end
    end
end

function update_dual_variables!(workset, trajectory)
    @unpack N = workset
    @unpack z_projection, z, α = workset.constraints
    @unpack w_projection, w, β = workset.constraints

    if z_projection !== nothing
        α .= α + trajectory.x[N+1] - z
    end

    if w_projection !== nothing
        for k in 1:N
            β[k] .= β[k] + trajectory.u[k] - w[k]
        end
    end

    return nothing
end
