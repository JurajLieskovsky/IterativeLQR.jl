# Terminal state constraint
struct Constraints{T}
    ρ::Ref{T}
    λN::Vector{T}

    function Constraints{T}(nx) where {T}
        new(zero(T), zeros(T, nx))
    end
end

## set functions

function set_penalty_parameter!(constraint::Constraints, ρ_new)
    @unpack ρ, λN = constraint
    λN .*= ρ[] / ρ_new
    ρ[] = ρ_new
    return nothing
end

## evaluation functions

function evaluate_penalties(constraint::Constraints, xN, xT)
    xT === nothing && return 0
    @unpack ρ, λN = constraint
    arg = xN - xT + λN
    return ρ[] / 2 * arg' * arg
end

function add_penalty_derivatives!(vxN, vxxN, constraint::Constraints, xN, xT)
    xT === nothing && return nothing
    @unpack ρ, λN = constraint
    vxN .+= ρ[] * (xN - xT + λN)
    vxxN[diagind(vxxN)] .+= ρ[]
    return nothing
end

function update_dual_variables!(constraint::Constraints, xN, xT)
    xT === nothing && return nothing
    @unpack λN = constraint
    λN .= λN + xN - xT
    return nothing
end
