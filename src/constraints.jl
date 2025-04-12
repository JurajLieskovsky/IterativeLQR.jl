# Terminal state constraint
struct Constraints{T}
    ρ::Ref{T}
    λN::Vector{T}

    function Constraints{T}(nx) where {T}
        new(zero(T), zeros(T, nx))
    end
end

## set functions

function set_penalty_parameter!(workset, ρ_new)
    @unpack ρ, λN = workset.constraints
    λN .*= ρ[] / ρ_new
    ρ[] = ρ_new
    return nothing
end

## general evaluation functions
evaluate_penalty(ρ, arg) = ρ / 2 * mapreduce(a -> a^2, +, arg)

function add_penalty_derivative!(grad, hess, ρ, arg)
    grad .+= ρ * arg
    hess[diagind(hess)] .+= ρ
end

# workset evaluation functions
function evaluate_penalties(workset, xN, xT)
    @unpack ρ, λN = workset.constraints
    return (xT !== nothing) ? evaluate_penalty(ρ[], xN - xT + λN) : 0
end

function add_penalty_derivatives!(workset, xN, xT)
    @unpack N = workset
    @unpack vx, vxx = workset.value_function
    @unpack ρ, λN = workset.constraints

    if xT !== nothing
        add_penalty_derivative!(vx[N+1], vxx[N+1], ρ[], xN - xT + λN)
    end

    return nothing
end

function update_dual_variables!(workset, xN, xT)
    @unpack λN = workset.constraints

    if xT !== nothing
        λN .= λN + xN - xT
    end

    return nothing
end
