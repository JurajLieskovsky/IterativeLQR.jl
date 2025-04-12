# Terminal state constraint
mutable struct Constraints{T}
    ρ::T

    zN_indicator::Union{Function,Nothing}
    zN::Vector{T}
    αN::Vector{T}

    # w::Vector{Vector{T}}
    # β::Vector{Vector{T}}

    function Constraints{T}(nx) where {T} # , nu, N) where {T}
        zN = zeros(T, nx)
        αN = zeros(T, nx)
        # β = [zeros(T, nu) for _ in 1:N]
        # w = [zeros(T, nu) for _ in 1:N]

        new(zero(T), nothing, αN, zN)#, β, w)
    end
end

## set functions

function set_terminal_constraint_function!(workset, fun)
    workset.constraints.zN_indicator = fun
end

function set_penalty_parameter!(workset, ρ_new)
    @unpack αN = workset.constraints

    ratio = workset.constraints.ρ / ρ_new
    workset.constraints.ρ = ρ_new

    αN .*= ratio

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
    @unpack ρ, zN_indicator, αN, zN = workset.constraints

    if zN_indicator !== nothing
        add_penalty_derivative!(vx[N+1], vxx[N+1], ρ, x[N+1] - zN + αN)
    end

    return nothing
end

function evaluate_penalties(workset, trajectory)
    @unpack N = workset
    @unpack ρ, zN_indicator, zN, αN = workset.constraints

    return (zN_indicator !== nothing) ? evaluate_penalty(ρ, trajectory.x[N+1] - zN + αN) : 0
end

function update_slack_variables!(workset, trajectory)
    @unpack N = workset
    @unpack zN_indicator, zN = workset.constraints

    if zN_indicator !== nothing
        zN .= zN_indicator(trajectory.x[N+1])
    end
end

function update_dual_variables!(workset, trajectory)
    @unpack N = workset
    @unpack zN_indicator, zN, αN = workset.constraints

    if zN_indicator !== nothing
        αN .= αN + trajectory.x[N+1] - zN
    end

    return nothing
end
