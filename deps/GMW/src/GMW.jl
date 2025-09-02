module GMW

using LinearAlgebra

struct MChol{T}
    p::Vector{Int}
    L::Matrix{T}
end

function factorize(A::AbstractMatrix{T}, δ=eps(T)) where {T}
    n, m = size(A)
    @assert n > 1
    @assert n == m

    # result allocation
    p = Vector{Int}(undef, n)
    L = Matrix{T}(undef, n, n)

    # inital copying
    p .= 1:n
    L .= A

    # calculation of β²
    γ = mapreduce(e -> abs(e), max, view(A, diagind(A)))
    ξ = mapreduce(e -> abs(e), max, LowerTriangular(view(A, 2:n, 1:n-1)))
    β2 = max(γ, ξ / sqrt(n^2 - 1), eps(T))

    for j in 1:n
        # find largest diagonal element in the block to be factorized
        q = j
        for i in j:n
            abs(L[i, i]) >= abs(L[q, q]) && (q = i)
        end

        # swap permutation
        p[q], p[j] = p[j], p[q]

        # swap rows
        for i in 1:n
            L[q, i], L[j, i] = L[j, i], L[q, i]
        end

        # swap cols
        for i in j:n
            L[i, q], L[i, j] = L[i, j], L[i, q]
        end

        # calculate factorization
        θ_j = j == n ? 0 : maximum(view(L, j+1:n, j))
        d_j = max(δ, abs(L[j, j]), θ_j^2 / β2)

        L[j, j] = sqrt(d_j)

        if j < n
            view(L, j, j+1:n) .= 0
            view(L, j+1:n, j) .= view(L, j+1:n, j) / L[j, j]
            view(L, j+1:n, j+1:n) .-= view(L, j+1:n, j) * view(L, j+1:n, j)'
        end
    end

    return MChol{T}(p, L)
end

function reconstruct!(A::AbstractMatrix{T}, F::MChol{T}) where {T}
    pL = view(F.L, invperm(F.p), 1:length(F.p))
    mul!(A, pL, pL')
    return nothing
end

function reconstruct(F::MChol)
    pL = view(F.L, invperm(F.p), 1:length(F.p))
    return pL * pL'
end

end # module GMW
