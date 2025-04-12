# Terminal state constraint
mutable struct Constraints{T}
    ρ::T

    z_indicator::Union{Function,Nothing}
    z::Vector{T}
    α::Vector{T}

    # w::Vector{Vector{T}}
    # β::Vector{Vector{T}}

    function Constraints{T}(nx) where {T} # , nu, N) where {T}
        z = zeros(T, nx)
        α = zeros(T, nx)
        # β = [zeros(T, nu) for _ in 1:N]
        # w = [zeros(T, nu) for _ in 1:N]

        new(zero(T), nothing, α, z)#, β, w)
    end
end

## set functions

function set_terminal_constraint_indicator_function!(workset, fun)
    workset.constraints.z_indicator = fun
end

function set_penalty_parameter!(workset, ρ_new)
    @unpack α = workset.constraints

    ratio = workset.constraints.ρ / ρ_new
    workset.constraints.ρ = ρ_new

    α .*= ratio

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
    @unpack x = nominal_trajectory(workset)
    @unpack vx, vxx = workset.value_function

    @unpack ρ = workset.constraints
    @unpack z_indicator, z, α = workset.constraints

    if z_indicator !== nothing
        add_penalty_derivative!(vx[N+1], vxx[N+1], ρ, x[N+1] - z + α)
    end

    return nothing
end

function evaluate_penalties(workset, trajectory)
    @unpack N = workset
    @unpack ρ = workset.constraints
    @unpack z_indicator, z, α = workset.constraints

    return (z_indicator !== nothing) ? evaluate_penalty(ρ, trajectory.x[N+1] - z + α) : 0
end

function update_slack_variables!(workset, trajectory)
    @unpack N = workset
    @unpack z_indicator, z, α = workset.constraints

    if z_indicator !== nothing
        z .= z_indicator(trajectory.x[N+1] + α)
    end
end

function update_dual_variables!(workset, trajectory)
    @unpack N = workset
    @unpack z_indicator, z, α = workset.constraints

    if z_indicator !== nothing
        α .= α + trajectory.x[N+1] - z
    end

    return nothing
end
