# Terminal state constraint

struct TerminalStateConstraint{T}
    ρ::Ref{T}
    λN::Vector{T}

    function TerminalStateConstraint{T}(nx) where {T}
        new(zero(T), zeros(T, nx))
    end
end

## set functions

function set_parameter!(constraint::TerminalStateConstraint, ρ_new)
    @unpack ρ, λN = constraint
    λN .*= ρ[] / ρ_new
    ρ[] = ρ_new
    return nothing
end

## evaluation functions

function evaluate(constraint::TerminalStateConstraint, xN, xT)
    @unpack ρ, λN = constraint
    arg = xN - xT + λN
    return ρ[] / 2 * arg' * arg
end

function add_derivatives!(vxN, vxxN, constraint::TerminalStateConstraint, xN, xT)
    @unpack ρ, λN = constraint
    vxN .+= ρ[] * (xN - xT + λN)
    vxxN[diagind(vxxN)] .+= ρ[]
    return nothing
end

function update_dual!(constraint::TerminalStateConstraint, xN, xT)
    @unpack λN = constraint
    λN .= λN + xN - xT
    return nothing
end
