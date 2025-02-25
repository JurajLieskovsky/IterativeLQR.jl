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
    Δv::Vector{Vector{T}}
    vx::Vector{Vector{T}}
    vxx::Vector{Matrix{T}}

    function ValueFunction{T}(ndx, N) where {T}
        Δv = [Vector{T}(undef, 2) for _ in 1:N+1]
        vx = [Vector{T}(undef, ndx) for _ in 1:N+1]
        vxx = [Matrix{T}(undef, ndx, ndx) for _ in 1:N+1]

        return new(Δv, vx, vxx)
    end
end

struct PolicyUpdate{T}
    d::Vector{Vector{T}}
    K::Vector{Matrix{T}}

    function PolicyUpdate{T}(ndx, nu, N) where {T}
        d = [Vector{T}(undef, nu) for _ in 1:N]
        K = [Matrix{T}(undef, nu, ndx) for _ in 1:N]

        return new(d, K)
    end
end

struct DynamicsDerivatives{T}
    jac::Vector{Matrix{T}}
    fx::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
    fu::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}

    function DynamicsDerivatives{T}(ndx, nu, N) where {T}
        jac = [Matrix{T}(undef, ndx, ndx + nu) for _ in 1:N]
        fx = [view(jac[k], 1:ndx, 1:ndx) for k in 1:N]
        fu = [view(jac[k], 1:ndx, ndx+1:ndx+nu) for k in 1:N]

        return new(jac, fx, fu)
    end
end

struct CostDerivatives{T}
    grad::Vector{Vector{T}}
    lx::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}}
    lu::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}}

    hess::Vector{Matrix{T}}
    lxx::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
    luu::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
    lux::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
    lxu::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}

    function CostDerivatives{T}(ndx, nu, N) where {T}
        grad = [Vector{T}(undef, ndx + nu) for _ in 1:N]
        lx = [view(grad[k], 1:ndx) for k in 1:N]
        lu = [view(grad[k], ndx+1:ndx+nu) for k in 1:N]

        hess = [Matrix{T}(undef, ndx + nu, ndx + nu) for _ in 1:N]
        lxx = [view(hess[k], 1:ndx, 1:ndx) for k in 1:N]
        luu = [view(hess[k], ndx+1:ndx+nu, ndx+1:ndx+nu) for k in 1:N]
        lux = [view(hess[k], ndx+1:ndx+nu, 1:ndx) for k in 1:N]
        lxu = [view(hess[k], 1:ndx, ndx+1:ndx+nu) for k in 1:N]

        return new(grad, lx, lu, hess, lxx, luu, lux, lxu)
    end
end

struct SubproblemObjectiveDerivatives{T}
    g::Vector{T}
    qx::SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}
    qu::SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}

    H::Matrix{T}
    qxx::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    quu::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    qux::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    qxu::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}

    function SubproblemObjectiveDerivatives{T}(ndx, nu) where {T}
        g = zeros(ndx + nu)
        qx = view(g, 1:ndx)
        qu = view(g, ndx+1:ndx+nu)

        H = zeros(ndx + nu, ndx + nu)
        qxx = view(H, 1:ndx, 1:ndx)
        quu = view(H, ndx+1:ndx+nu, ndx+1:ndx+nu)
        qux = view(H, ndx+1:ndx+nu, 1:ndx)
        qxu = view(H, 1:ndx, ndx+1:ndx+nu)

        return new(g, qx, qu, H, qxx, quu, qux, qxu)
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
    subproblem_objective_derivatives::SubproblemObjectiveDerivatives{T}

    function Workset{T}(nx, nu, N, ndx=nothing) where {T}
        ndx = ndx !== nothing ? ndx : nx

        trajectory = (Trajectory{T}(nx, nu, N), Trajectory{T}(nx, nu, N))
        value_function = ValueFunction{T}(ndx, N)
        policy_update = PolicyUpdate{T}(ndx, nu, N)
        dynamics_derivatives = DynamicsDerivatives{T}(ndx, nu, N)
        cost_derivatives = CostDerivatives{T}(ndx, nu, N)

        subproblem_objective_derivatives = SubproblemObjectiveDerivatives{T}(ndx, nu)

        return new(N, nx, ndx, nu, 1, 2, trajectory, value_function, policy_update, dynamics_derivatives, cost_derivatives, subproblem_objective_derivatives)
    end
end

