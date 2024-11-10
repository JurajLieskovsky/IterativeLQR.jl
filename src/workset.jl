struct Trajectory{T}
    x::Vector{Vector{T}}
    u::Vector{Vector{T}}
    l::Vector{T}

    function Trajectory{T}(nx, nu, N) where {T}
        x = [zeros(T, nx) for _ in 1:N+1]
        u = [zeros(T, nu) for _ in 1:N]
        l = zeros(T, N + 1)

        return new(x, u, l)
    end
end

struct ValueFunction{T}
    Δv::Vector{T}
    vx::Vector{Vector{T}}
    vxx::Vector{Matrix{T}}

    function ValueFunction{T}(ndx, N) where {T}
        Δv = Vector{T}(undef, N)
        vx = [Vector{T}(undef, ndx) for _ in 1:N+1]
        vxx = [Matrix{T}(undef, ndx, ndx) for _ in 1:N+1]

        return new(Δv, vx, vxx)
    end
end

struct PolicyUpdate{T}
    Δu::Vector{Vector{T}}
    Δux::Vector{Matrix{T}}

    function PolicyUpdate{T}(ndx, nu, N) where {T}
        Δu = [Vector{T}(undef, nu) for _ in 1:N]
        Δux = [Matrix{T}(undef, nu, ndx) for _ in 1:N]

        return new(Δu, Δux)
    end
end

struct DynamicsDerivatives{T}
    fx::Vector{Matrix{T}}
    fu::Vector{Matrix{T}}

    function DynamicsDerivatives{T}(ndx, nu, N) where {T}
        fx = [Matrix{T}(undef, ndx, ndx) for _ in 1:N]
        fu = [Matrix{T}(undef, ndx, nu) for _ in 1:N]

        return new(fx, fu)
    end
end

struct CostDerivatives{T}
    lx::Vector{Vector{T}}
    lu::Vector{Vector{T}}

    lxx::Vector{Matrix{T}}
    lxu::Vector{Matrix{T}}
    luu::Vector{Matrix{T}}

    function CostDerivatives{T}(ndx, nu, N) where {T}
        lx = [Vector{T}(undef, ndx) for _ in 1:N]
        lu = [Vector{T}(undef, nu) for _ in 1:N]

        lxx = [Matrix{T}(undef, ndx, ndx) for _ in 1:N]
        lxu = [Matrix{T}(undef, ndx, nu) for _ in 1:N]
        luu = [Matrix{T}(undef, nu, nu) for _ in 1:N]

        return new(lx, lu, lxx, lxu, luu)
    end
end

struct Workset{T}
    N::Int64
    nx::Int64
    ndx::Int64
    nu::Int64
    nominal::Ref{Int}
    active::Ref{Int}
    trajectory::Tuple{Trajectory{T},Trajectory{T}}
    value_function::ValueFunction{T}
    policy_update::PolicyUpdate{T}
    dynamics_derivatives::DynamicsDerivatives{T}
    cost_derivatives::CostDerivatives{T}

    function Workset{T}(nx, nu, N, ndx=nothing) where {T}
        ndx = ndx !== nothing ? ndx : nx

        trajectory = (Trajectory{T}(nx, nu, N), Trajectory{T}(nx, nu, N))
        value_function = ValueFunction{T}(ndx, N)
        policy_update = PolicyUpdate{T}(ndx, nu, N)
        dynamics_derivatives = DynamicsDerivatives{T}(ndx, nu, N)
        cost_derivatives = CostDerivatives{T}(ndx, nu, N)

        return new(N, nx, ndx, nu, 1, 2, trajectory, value_function, policy_update, dynamics_derivatives, cost_derivatives)
    end
end

