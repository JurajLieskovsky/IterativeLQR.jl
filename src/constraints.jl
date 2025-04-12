# Terminal state constraint

struct TerminalStateConstraint{T}
    ρ::Ref{T}
    xN::Vector{T}
    λN::Vector{T}

    function TerminalStateConstraint{T}(nx) where {T}
        new(zero(T), zeros(T, nx), zeros(T, nx))
    end
end

## set functions

function set_parameter!(constraint::TerminalStateConstraint, ρ_new)
    @unpack ρ, λN = constraint

    λN .*= ρ[] / ρ_new
    ρ[] = ρ_new

    return nothing
end

function set_terminal_state!(constraint::TerminalStateConstraint, xN)
    constraint.xN .= xN
    return nothing
end

## evaluation functions

function evaluate(constraint::TerminalStateConstraint, state)
    @unpack ρ, xN, λN = constraint
    arg = state - xN + λN
    return ρ[] / 2 * arg' * arg
end

function add_derivatives!(vxN, vxxN, constraint::TerminalStateConstraint, state)
    @unpack ρ, xN, λN = constraint
    vxN .+= ρ[] * (state - xN + λN)
    vxxN[diagind(vxxN)] .+= ρ[]
    return nothing
end

function update_dual!(constraint::TerminalStateConstraint, state)
    @unpack xN, λN = constraint
    λN .= λN + state - xN
    return nothing
end
