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

struct CoordinateJacobians{T}
    aug_E::Vector{Matrix{T}}
    E # TBD

    function CoordinateJacobians{T}(nx, ndx, nu, N) where {T}
        aug_E = [zeros(T, nx + nu, ndx + nu) for _ in 1:N]
        E = [k <= N ? view(aug_E[k], 1:nx, 1:ndx) : zeros(T, nx, ndx) for k in 1:N+1]

        for i in 1:N
            aug_E[i][nx+1:nx+nu, ndx+1:ndx+nu] .= Matrix{T}(I, nu, nu)
        end
        
        return new(aug_E, E)
    end
end

struct DynamicsDerivatives{T}
    ∇f::Vector{Matrix{T}}
    fx::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
    fu::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}

    ∇2f::Vector{Array{T,3}}
    fxx::Vector{SubArray{T,3,Array{T,3},Tuple{UnitRange{Int64},UnitRange{Int64},UnitRange{Int64}},false}}
    fuu::Vector{SubArray{T,3,Array{T,3},Tuple{UnitRange{Int64},UnitRange{Int64},UnitRange{Int64}},false}}
    fux::Vector{SubArray{T,3,Array{T,3},Tuple{UnitRange{Int64},UnitRange{Int64},UnitRange{Int64}},false}}
    fxu::Vector{SubArray{T,3,Array{T,3},Tuple{UnitRange{Int64},UnitRange{Int64},UnitRange{Int64}},false}}

    function DynamicsDerivatives{T}(nx, nu, N) where {T}
        ∇f = [Matrix{T}(undef, nx, nx + nu) for _ in 1:N]
        fx = [view(∇f[k], 1:nx, 1:nx) for k in 1:N]
        fu = [view(∇f[k], 1:nx, nx+1:nx+nu) for k in 1:N]

        ∇2f = [Array{T,3}(undef, nx, nx + nu, nx + nu) for _ in 1:N]
        fxx = [view(∇2f[k], 1:nx, 1:nx, 1:nx) for k in 1:N]
        fuu = [view(∇2f[k], 1:nx, nx+1:nx+nu, nx+1:nx+nu) for k in 1:N]
        fux = [view(∇2f[k], 1:nx, nx+1:nx+nu, 1:nx) for k in 1:N]
        fxu = [view(∇2f[k], 1:nx, 1:nx, nx+1:nx+nu) for k in 1:N]

        return new(∇f, fx, fu, ∇2f, fxx, fuu, fux, fxu)
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
    coordinate_jacobians::CoordinateJacobians{T}
    dynamics_derivatives::DynamicsDerivatives{T}
    cost_derivatives::CostDerivatives{T}
    subproblem_objective_derivatives::SubproblemObjectiveDerivatives{T}

    function Workset{T}(nx, nu, N, ndx=nothing) where {T}
        ndx = ndx !== nothing ? ndx : nx

        trajectory = (Trajectory{T}(nx, nu, N), Trajectory{T}(nx, nu, N))
        value_function = ValueFunction{T}(ndx, N)
        policy_update = PolicyUpdate{T}(ndx, nu, N)
        coordinate_jacobians = CoordinateJacobians{T}(nx, ndx, nu, N)
        dynamics_derivatives = DynamicsDerivatives{T}(nx, nu, N)
        cost_derivatives = CostDerivatives{T}(nx, nu, N)

        subproblem_objective_derivatives = SubproblemObjectiveDerivatives{T}(ndx, nu)

        return new(N, nx, ndx, nu, 1, 2, trajectory, value_function, policy_update, coordinate_jacobians, dynamics_derivatives, cost_derivatives, subproblem_objective_derivatives)
    end
end

