module RungeKutta

using LinearAlgebra

abstract type Method end

struct ExplicitMethod <: Method
    s
    A
    b
    c
end

struct ImplicitMethod <: Method
    s
    A
    b
    c
end

function Method(A, b, c)
    s = length(c)

    @assert length(b) == s
    @assert size(A) == (s, s)

    explicit = all([A[j, i] == 0 for j in 1:s for i in j:s])

    if explicit
        return ExplicitMethod(s, A, b, c)
    else
        return ImplicitMethod(s, A, b, c)
    end
end

function f!(xnew, m::ExplicitMethod, F!, x, u, h)
    # matrix for storing stages
    K = Matrix{eltype(x)}(undef, length(x), m.s)

    # first stage
    @views F!(K[:, 1], x, u)

    # remaining stages (xnew is used for storing x_j)
    for j in 2:m.s
        xnew .= x
        @views mul!(xnew, K[:, 1:j-1], m.A[j, 1:j-1], h, 1)
        @views F!(K[:, j], xnew, u)
    end

    # new state
    xnew .= x
    mul!(xnew, K, m.b, h, 1)

    return nothing
end

#= function step_diff!(jac, m::ExplicitMethod, dF!, x, u, h)
    # arrays for storing stage jacobians
    fx = Array{eltype(x),3}(undef, length(x), length(x), m.s)
    fu = Array{eltype(x),3}(undef, length(x), length(u), m.s)
end =#

end

