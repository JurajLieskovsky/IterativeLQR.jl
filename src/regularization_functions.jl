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
    λ, V = try
        eigen(Symmetric(H))
    catch e
        @warn("Eigen value decompostion failed during regularization with $e")
        return nothing
    end
    λ_reg = map(e -> e < δ ? δ : e, λ)
    H .= V * diagm(λ_reg) * V'
    return nothing
end

function gmw_regularization!(H, δ)
    F = GMW.factorize(H, δ)
    GMW.reconstruct!(H, F)
    return nothing
end
