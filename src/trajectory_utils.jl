function Base.copy!(dst::Trajectory, src::Trajectory)
    @assert size(dst.x) == size(src.x)
    @assert size(dst.u) == size(src.u)
    @assert size(dst.l) == size(src.l)

    for i in 1:length(src.x)
        dst.x[i] .= src.x[i]
    end

    for i in 1:length(src.u)
        dst.u[i] .= src.u[i]
    end

    dst.l .= src.l
end

## nominal/active utilities

function nominal_trajectory(workset)
    workset.trajectory[workset.nominal[]]
end

function active_trajectory(workset)
    workset.trajectory[workset.active[]]
end

function swap_trajectories!(workset)
    workset.nominal[], workset.active[] = workset.active[], workset.nominal[]
    return nothing
end

## set functions

function set_initial_state!(workset, x0)
    nominal_trajectory(workset).x[1] .= x0
end

function set_initial_inputs!(workset, u::Vector{Vector{T}}) where {T}
    @assert length(u) == workset.N

    for k in 1:workset.N
        nominal_trajectory(workset).u[k] .= u[k]
    end
end

function set_initial_inputs!(workset, u::Matrix{T}) where {T}
    @unpack N, nu = workset

    (p, q) = size(u)

    if p == N && q == nu
        iterator = eachrow(u)
    elseif p == nu && q == N
        iterator = eachcol(u)
    else
        error("dimensions do not match horizon's length and/or number of inputs")
    end

    for (k, input) in enumerate(iterator)
        nominal_trajectory(workset).u[k] .= input
    end
end
