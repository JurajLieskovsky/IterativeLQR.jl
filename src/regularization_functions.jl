"""
Regularizes the symmetric matrix `H` so that it is positive definite. The argument `approach`
determines if an eigenvalue approach is taken [`eigenvalue_regularization!`](@ref) or the GMW
modified Cholesky decomposition algorithm is used [`gmw_regularization!`](@ref).

"""
function regularize!(H::AbstractMatrix, δ::Real, approach::Symbol)
    if approach == :eig
        eigenvalue_regularization!(H, δ)
    elseif approach == :mchol
        gmw_regularization!(H, δ)
    else
        error("undefined regularization approach")
    end
end

"""
Regularizes `H` using the eigenvalue approach, so that all of its eigen values are >=`δ`.

"""
function eigenvalue_regularization!(H, δ::Real)
    @assert δ > 0

    # remove potential asymetries
    H .+= H'
    H ./= 2

    # calculate eigenvalues and eigenvectors
    λ, _, _, V = LinearAlgebra.LAPACK.geev!('N', 'V', H)

    # minimally perturb eigenvalues and reconstruct matrix
    map!(e -> e < δ ? δ : e, λ, λ)
    H .= V * Diagonal(λ) * V'

    return nothing
end

"""
Regularized `H` using the Gill-Murray-Wright modified Cholesky factorization.

"""
function gmw_regularization!(H, δ)
    p, L = GMW81.factorize(H, δ)
    GMW81.reconstruct!(H, p, L)
    return nothing
end
