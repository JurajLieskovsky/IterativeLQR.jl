function regularize!(H, δ, approach)
    if approach == :eig
        eigenvalue_regularization!(H, δ)
    elseif approach == :mchol
        gmw_regularization!(H, δ)
    else
        error("undefined regularization approach")
    end
end

function eigenvalue_regularization!(H, δ)
    # remove potential asymetries
    H .+= H'
    H ./= 2

    # calculate eigenvalues and eigenvectors
    #  imaginary components of eigenvalues can be ignored
    #  as the matrix is symmetric
    λ, _, _, V = LinearAlgebra.LAPACK.geev!('N', 'V', H)

    # minimally perturb eigenvalues and reconstruct matrix
    map!(e -> e < δ ? δ : e, λ, λ)
    H .= V * Diagonal(λ) * V'

    return nothing
end

function gmw_regularization!(H, δ)
    # perform gill-murray-wright modified Cholesky factorization
    #  assumes the matrix is symmetric and only uses lower part
    F = GMW.factorize(H, δ)

    # reconstruct the now positive-definite matrix
    GMW.reconstruct!(H, F)

    return nothing
end
