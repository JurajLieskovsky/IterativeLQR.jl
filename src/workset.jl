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

struct PolicyUpdate{T}
    d::Vector{Vector{T}}
    K::Vector{Matrix{T}}

    function PolicyUpdate{T}(ndx, nu, N) where {T}
        d = [Vector{T}(undef, nu) for _ in 1:N]
        K = [Matrix{T}(undef, nu, ndx) for _ in 1:N]

        return new(d, K)
    end
end

struct CoordinateJacobians{T}
    E::Vector{Matrix{T}}

    function CoordinateJacobians{T}(nx, ndx, N) where {T}
        E = [zeros(T, nx, ndx) for _ in 1:N+1]

        return new(E)
    end
end

struct DynamicsDerivatives{T}
    ∇f::Vector{Matrix{T}}
    fx::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
    fu::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}

    function DynamicsDerivatives{T}(nx, nu, N) where {T}
        ∇f = [Matrix{T}(undef, nx, nx + nu) for _ in 1:N]
        fx = [view(∇f[k], 1:nx, 1:nx) for k in 1:N]
        fu = [view(∇f[k], 1:nx, nx+1:nx+nu) for k in 1:N]

        return new(∇f, fx, fu)
    end
end

struct CostDerivatives{T}
    ∇l::Vector{Vector{T}}
    lx::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}}
    lu::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}}

    ∇2l::Vector{Matrix{T}}
    lxx::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
    luu::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
    lux::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
    lxu::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}

    Φx::Vector{T}
    Φxx::Matrix{T}

    function CostDerivatives{T}(nx, nu, N) where {T}
        ∇l = [Vector{T}(undef, nx + nu) for _ in 1:N]
        lx = [view(∇l[k], 1:nx) for k in 1:N]
        lu = [view(∇l[k], nx+1:nx+nu) for k in 1:N]

        ∇2l = [Matrix{T}(undef, nx + nu, nx + nu) for _ in 1:N]
        lxx = [view(∇2l[k], 1:nx, 1:nx) for k in 1:N]
        luu = [view(∇2l[k], nx+1:nx+nu, nx+1:nx+nu) for k in 1:N]
        lux = [view(∇2l[k], nx+1:nx+nu, 1:nx) for k in 1:N]
        lxu = [view(∇2l[k], 1:nx, nx+1:nx+nu) for k in 1:N]

        Φx = Vector{T}(undef, nx)
        Φxx = Matrix{T}(undef, nx, nx)

        return new(∇l, lx, lu, ∇2l, lxx, luu, lux, lxu, Φx, Φxx)
    end
end

struct BackwardPassWorkset{T}
    Δv::Ref{T}
    vx::Vector{T}
    vxx::Matrix{T}

    g::Vector{T}
    qx::SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}
    qu::SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}

    H::Matrix{T}
    qxx::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    quu::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    qux::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    qxu::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}

    function BackwardPassWorkset{T}(ndx, nu) where {T}
        Δv = zero(T)
        vx = zeros(T, ndx)
        vxx = zeros(T, ndx, ndx)

        g = zeros(T, ndx + nu)
        qx = view(g, 1:ndx)
        qu = view(g, ndx+1:ndx+nu)

        H = zeros(T, ndx + nu, ndx + nu)
        qxx = view(H, 1:ndx, 1:ndx)
        quu = view(H, ndx+1:ndx+nu, ndx+1:ndx+nu)
        qux = view(H, ndx+1:ndx+nu, 1:ndx)
        qxu = view(H, 1:ndx, ndx+1:ndx+nu)

        return new(Δv, vx, vxx, g, qx, qu, H, qxx, quu, qux, qxu)
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
    policy_update::PolicyUpdate{T}
    coordinate_jacobians::CoordinateJacobians{T}
    dynamics_derivatives::DynamicsDerivatives{T}
    cost_derivatives::CostDerivatives{T}
    tangent_dynamics_derivatives::DynamicsDerivatives{T}
    tangent_cost_derivatives::CostDerivatives{T}
    backward_pass_workset::BackwardPassWorkset{T}

    function Workset{T}(nx, nu, N, ndx=nothing) where {T}
        ndx = ndx !== nothing ? ndx : nx

        trajectory = (Trajectory{T}(nx, nu, N), Trajectory{T}(nx, nu, N))
        policy_update = PolicyUpdate{T}(ndx, nu, N)
        coordinate_jacobians = CoordinateJacobians{T}(nx, ndx, N)
        dynamics_derivatives = DynamicsDerivatives{T}(nx, nu, N)
        cost_derivatives = CostDerivatives{T}(nx, nu, N)
        tangent_dynamics_derivatives = DynamicsDerivatives{T}(ndx, nu, N)
        tangent_cost_derivatives = CostDerivatives{T}(ndx, nu, N)

        backward_pass_workset = BackwardPassWorkset{T}(ndx, nu)

        return new(N, nx, ndx, nu, 1, 2, trajectory, policy_update, coordinate_jacobians, dynamics_derivatives, cost_derivatives, tangent_dynamics_derivatives, tangent_cost_derivatives, backward_pass_workset)
    end
end

