"""
Struct containing a trajectory xₖ, uₖ of the controlled system (including costs).

"""
struct Trajectory{T}
    x::Vector{Vector{T}}
    u::Vector{Vector{T}}
    l::Vector{T}

    function Trajectory{T}(nx::Int, nu::Int, N::Int) where {T}
        x = [zeros(T, nx) for _ in 1:N+1]
        u = [zeros(T, nu) for _ in 1:N]
        l = zeros(T, N + 1)

        return new(x, u, l)
    end
end

"""
Struct containing terms dₖ and Kₖ of the policy update δuₖ(δxₖ) = dₖ + Kₖδxₖ.

"""
struct PolicyUpdate{T}
    d::Vector{Vector{T}}
    K::Vector{Matrix{T}}

    function PolicyUpdate{T}(ndx::Int, nu::Int, N::Int) where {T}
        d = [Vector{T}(undef, nu) for _ in 1:N]
        K = [Matrix{T}(undef, nu, ndx) for _ in 1:N]

        return new(d, K)
    end
end

"""
Struct containing coordinate jacobians Eₖ.

"""
struct CoordinateJacobians{T}
    E::Vector{Matrix{T}}

    function CoordinateJacobians{T}(nx::Int, ndx::Int, N::Int) where {T}
        E = [zeros(T, nx, ndx) for _ in 1:N+1]

        return new(E)
    end
end

"""
Struct containing partial derivatives of the system's dyanamics f.

"""
struct DynamicsDerivatives{T}
    ∇f::Vector{Matrix{T}}
    fx::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
    fu::Vector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Int64},UnitRange{Int64}},false}}

    function DynamicsDerivatives{T}(nx::Int, nu::Int, N::Int) where {T}
        ∇f = [Matrix{T}(undef, nx, nx + nu) for _ in 1:N]
        fx = [view(∇f[k], 1:nx, 1:nx) for k in 1:N]
        fu = [view(∇f[k], 1:nx, nx+1:nx+nu) for k in 1:N]

        return new(∇f, fx, fu)
    end
end

"""
Struct containing partial derivatives of the running cost l and the final cost Φ.

"""
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

"""
Struct containing quantities used exclusively in the backward pass of the algorithm.

"""
struct BackwardPassWorkset{T}
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

    function BackwardPassWorkset{T}(ndx::Int, nu::Int) where {T}
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

        return new(vx, vxx, g, qx, qu, H, qxx, quu, qux, qxu)
    end
end

"""
Struct containing the nominal trajectory (current the most optimal) as well as all internal quantities of the algorithm.

"""
struct Workset{T}
    N::Int64
    nx::Int64
    ndx::Int64
    nu::Int64
    nominal::Ref{Int}
    active::Ref{Int}

    trajectory::Tuple{Trajectory{T},Trajectory{T}}
    policy_update::PolicyUpdate{T}
    dynamics_derivatives::DynamicsDerivatives{T}
    cost_derivatives::CostDerivatives{T}
    backward_pass_workset::BackwardPassWorkset{T}

    coordinate_jacobians::Union{CoordinateJacobians{T}, Nothing}
    dynamics_derivatives_workset::Union{DynamicsDerivatives{T}, Nothing}
    cost_derivatives_workset::Union{CostDerivatives{T}, Nothing}

    @doc """
    Create a `Workset` for the trajectory optimization problem.

    # Arguments
    - `nx`: number of states
    - `nu`: number of inputs
    - `N`: length of the optimizations horizon 
    - `ndx`: number of independent states (defaults to `nx`)

    # Returns

    """
    function Workset{T}(nx::Int, nu::Int, N::Int, ndx::Int=nx) where {T}

        trajectory = (Trajectory{T}(nx, nu, N), Trajectory{T}(nx, nu, N))
        policy_update = PolicyUpdate{T}(ndx, nu, N)
        dynamics_derivatives = DynamicsDerivatives{T}(ndx, nu, N)
        cost_derivatives = CostDerivatives{T}(ndx, nu, N)
        backward_pass_workset = BackwardPassWorkset{T}(ndx, nu)

        coordinate_jacobians = ndx != nx ? CoordinateJacobians{T}(nx, ndx, N) : nothing
        dynamics_derivatives_workset = ndx != nx ? DynamicsDerivatives{T}(nx, nu, nthreads()) : nothing
        cost_derivatives_workset = ndx != nx ? CostDerivatives{T}(nx, nu, nthreads()) : nothing

        return new(
            N, nx, ndx, nu, 1, 2,
            trajectory, policy_update,
            dynamics_derivatives, cost_derivatives, backward_pass_workset,
            coordinate_jacobians, dynamics_derivatives_workset, cost_derivatives_workset,
        )
    end
end


