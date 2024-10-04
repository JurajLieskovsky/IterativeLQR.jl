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

struct CostToGo{T}
    Δv::Vector{T}
    vx::Vector{Vector{T}}
    vxx::Vector{Matrix{T}}

    function CostToGo{T}(nx, N) where {T}
        Δv = Vector{T}(undef, N)
        vx = [Vector{T}(undef, nx) for _ in 1:N+1]
        vxx = [Matrix{T}(undef, nx, nx) for _ in 1:N+1]

        return new(Δv, vx, vxx)
    end
end

struct PolicyUpdate{T}
    Δu::Vector{Vector{T}}
    Δux::Vector{Matrix{T}}

    function PolicyUpdate{T}(nx, nu, N) where {T}
        Δu = [Vector{T}(undef, nu) for _ in 1:N]
        Δux = [Matrix{T}(undef, nu, nx) for _ in 1:N]

        return new(Δu, Δux)
    end
end

struct DynamicsDerivatives{T}
    fx::Vector{Matrix{T}}
    fu::Vector{Matrix{T}}

    function DynamicsDerivatives{T}(nx, nu, N) where {T}
        fx = [Matrix{T}(undef, nx, nx) for _ in 1:N]
        fu = [Matrix{T}(undef, nx, nu) for _ in 1:N]

        return new(fx, fu)
    end
end

struct CostDerivatives{T}
    lx::Vector{Vector{T}}
    lu::Vector{Vector{T}}

    lxx::Vector{Matrix{T}}
    lxu::Vector{Matrix{T}}
    luu::Vector{Matrix{T}}

    function CostDerivatives{T}(nx, nu, N) where {T}
        lx = [Vector{T}(undef, nx) for _ in 1:N]
        lu = [Vector{T}(undef, nu) for _ in 1:N]

        lxx = [Matrix{T}(undef, nx, nx) for _ in 1:N]
        lxu = [Matrix{T}(undef, nx, nu) for _ in 1:N]
        luu = [Matrix{T}(undef, nu, nu) for _ in 1:N]

        return new(lx, lu, lxx, lxu, luu)
    end
end

struct Workset{T}
    N::Int64
    nx::Int64
    nu::Int64
    nominal::Ref{Int}
    active::Ref{Int}
    trajectory::Tuple{Trajectory{T},Trajectory{T}}
    cost_to_go::CostToGo{T}
    policy_update::PolicyUpdate{T}
    dynamics_derivatives::DynamicsDerivatives{T}
    cost_derivatives::CostDerivatives{T}

    function Workset{T}(nx, nu, N) where {T}
        trajectory = (Trajectory{T}(nx, nu, N), Trajectory{T}(nx, nu, N))
        cost_to_go = CostToGo{T}(nx, N)
        policy_update = PolicyUpdate{T}(nx, nu, N)
        dynamics_derivatives = DynamicsDerivatives{T}(nx, nu, N)
        cost_derivatives = CostDerivatives{T}(nx, nu, N)

        return new(N, nx, nu, 1, 2, trajectory, cost_to_go, policy_update, dynamics_derivatives, cost_derivatives)
    end
end

