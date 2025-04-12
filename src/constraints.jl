# Terminal state constraint
struct Constraints{T}
    ρ::Ref{T}
    αN::Vector{T}

    function Constraints{T}(nx) where {T}
        new(zero(T), zeros(T, nx))
    end
end

## set functions

function set_penalty_parameter!(workset, ρ_new)
    @unpack ρ, αN = workset.constraints
    αN .*= ρ[] / ρ_new
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
function add_penalty_derivatives!(workset, xT)
    @unpack N = workset
    @unpack x = nominal_trajectory(workset)
    @unpack vx, vxx = workset.value_function
    @unpack ρ, αN = workset.constraints

    if xT !== nothing
        add_penalty_derivative!(vx[N+1], vxx[N+1], ρ[], x[N+1] - xT + αN)
    end

    return nothing
end

function evaluate_penalties(workset, trajectory, xT)
    @unpack N = workset
    @unpack ρ, αN = workset.constraints

    return (xT !== nothing) ? evaluate_penalty(ρ[], trajectory.x[N+1] - xT + αN) : 0
end

function update_dual_variables!(workset, trajectory, xT)
    @unpack N = workset
    @unpack αN = workset.constraints
    
    if xT !== nothing
        αN .= αN + trajectory.x[N+1] - xT
    end

    return nothing
end
