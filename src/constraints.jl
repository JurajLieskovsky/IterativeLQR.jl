# Terminal state constraint
struct Constraints{T}
    ρ::Ref{T}
    _xN::Ref{Bool}
    αN::Vector{T}
    zN::Vector{T}

    function Constraints{T}(nx) where {T} # , nu, N) where {T}
        αN = zeros(T, nx)
        zN = zeros(T, nx)
        new(zero(T), true, αN, zN)#, β, w)
    end
end

## set functions

function set_penalty_parameter!(workset, ρ_new)
    @unpack ρ, αN = workset.constraints

    ratio = ρ[] / ρ_new
    ρ[] = ρ_new

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
    @unpack ρ, _xN, αN, zN = workset.constraints

    if _xN[]
        add_penalty_derivative!(vx[N+1], vxx[N+1], ρ[], x[N+1] - zN + αN)
    end

    return nothing
end

function evaluate_penalties(workset, trajectory)
    @unpack N = workset
    @unpack ρ, _xN, zN, αN = workset.constraints

    return _xN[] ? evaluate_penalty(ρ[], trajectory.x[N+1] - zN + αN) : 0
end

function update_slack_variables!(workset, trajectory, terminal_constraint)
    @unpack N = workset
    @unpack _xN, zN = workset.constraints

    if terminal_constraint !== nothing
        _xN[] = true
        zN .= terminal_constraint(trajectory.x[N+1])
    else
        _xN[] = false
    end
end

function update_dual_variables!(workset, trajectory)
    @unpack N = workset
    @unpack _xN, zN, αN = workset.constraints

    if _xN[]
        αN .= αN + trajectory.x[N+1] - zN
    end

    return nothing
end
