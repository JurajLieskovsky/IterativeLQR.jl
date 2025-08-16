function min_regularization!(H, δ)
    λ, V = eigen(Symmetric(H))
    λ_reg = map(e -> e < δ ? δ : e, λ)
    H .= V * diagm(λ_reg) * V'
    return nothing
end

function bunchkaufman_regularization!(H, δ)
    S = bunchkaufman(Symmetric(H, :L))
    λ, V = eigen(S.D)
    λ_reg = map(e -> e < δ ? δ : e, λ)
    H .= V * diagm(λ_reg) * V'
    view(H, S.p, S.p) .= S.L * H * S.L'
    return nothing
end

function flip_regularization!(H, δ)
    λ, V = eigen(Symmetric(H))
    λ_reg = map(e -> e < δ ? max(δ, -e) : e, λ)
    H .= V * diagm(λ_reg) * V'
    return nothing
end

function holy_regularization!(H)
    # the approach is frankly **** for some PSD matrices
    # therefore the factorization is run only for benchmarking
    _ = cholesky(Positive, H)
    # H .= F.L * F.L'
    return nothing
end

