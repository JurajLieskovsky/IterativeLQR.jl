function regularize!(H, δ, approach)
    if approach == :eig
        eigenvalue_regularization!(H, δ)
    elseif approach == :gmw
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
    λ, _, _, V = LinearAlgebra.LAPACK.geev!('N', 'V', H)

    # minimally perturb eigenvalues and reconstruct matrix
    map!(e -> e < δ ? δ : e, λ, λ)
    H .= V * Diagonal(λ) * V'

    return nothing
end

function gmw_regularization!(H, δ)
    F = GMW.factorize(H, δ)
    GMW.reconstruct!(H, F)
    return nothing
end
