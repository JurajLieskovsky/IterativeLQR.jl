# ADMM constraints
struct ADMMConstraint{T}
    ρ::Matrix{T} # penalty parameter
    z::Vector{T} # slack variable
    α::Vector{T} # dual variable

    r::Vector{T} # primal residual
    s::Vector{T} # dual residual

    function ADMMConstraint{T}(n) where {T}
        ρ = Matrix{T}(I, n, n)
        z = zeros(T, n)
        α = zeros(T, n)
        r = zeros(T, n)
        s = zeros(T, n)
        return new(ρ, z, α, r, s)
    end
end

mutable struct Constraints{T}
    terminal_state_projection::Union{Function,Nothing}
    terminal_state_constraint::ADMMConstraint{T}

    input_projection::Union{Function,Nothing}
    input_constraint::Vector{ADMMConstraint{T}}

    state_projection::Union{Function,Nothing}
    state_constraint::Vector{ADMMConstraint{T}}

    function Constraints{T}(nx, nu, N) where {T}
        terminal_state_projection = nothing
        terminal_state_constraint = ADMMConstraint{T}(nx)

        input_projection = nothing
        input_constraint = [ADMMConstraint{T}(nu) for _ in 1:N]

        state_projection = nothing
        state_constraint = [ADMMConstraint{T}(nx) for _ in 1:N+1]

        return new(terminal_state_projection, terminal_state_constraint, input_projection, input_constraint, state_projection, state_constraint)
    end
end

## projection setting functions
function set_terminal_state_projection_function!(workset, Π::Function)
    setproperty!(workset.constraints, :terminal_state_projection, Π)
end

function set_input_projection_function!(workset, Π::Function)
    setproperty!(workset.constraints, :input_projection, Π)
end

function set_state_projection_function!(workset, Π::Function)
    setproperty!(workset.constraints, :state_projection, Π)
end

## penalty parameter setting functions
function set_constraint_parameter!(constraint, ρ_new)
    @unpack α, ρ = constraint
    α .= inv(ρ_new) * ρ * α
    α .= map(α_k -> isnan(α_k) ? 0 : α_k, α)
    ρ .= ρ_new
    return nothing
end

function set_terminal_state_constraint_parameter!(workset, ρ_new)
    set_constraint_parameter!(workset.constraints.terminal_state_constraint, ρ_new)
    return nothing
end

function set_input_constraint_parameter!(workset, ρ_new)
    ThreadsX.map(constraint -> set_constraint_parameter!(constraint, ρ_new), workset.constraints.input_constraint)
    return nothing
end

function set_state_constraint_parameter!(workset, ρ_new)
    ThreadsX.map(constraint -> set_constraint_parameter!(constraint, ρ_new), workset.constraints.state_constraint)
    return nothing
end

## evaluation functions
function add_penalty_derivative!(gradient, hessian, constraint, primal)
    @unpack ρ, z, α = constraint
    gradient .+= ρ * (primal - z + α)
    hessian .+= ρ
    return nothing
end

function evaluate_penalty(constraint, primal)
    @unpack ρ, z, α = constraint
    arg = primal - z + α
    return 0.5 * arg' * ρ * arg
end

function mul_update_penalty_parameter(ρ, r, s, ϵ=1e-3, ρ_max=1e8, ρ_min=1e-8)
    R = r * r' 
    R[diagind(R)] .+= ϵ

    S = s * s' 
    S[diagind(S)] .+= ϵ

    M = R * inv(S)

    ρ_new = ρ * exp(0.02 * log(M))

    λ, V = eigen(Symmetric(ρ_new))
    λ_reg = map(e -> e < ρ_min ? ρ_min : (e > ρ_max ? ρ_max : e), λ)
    return V * diagm(λ_reg) * V'
end

function add_update_penalty_parameter(ρ, r, s, α=0.1, ρ_max=1e8, ρ_min=1e-8)
    ρ_new = ρ + α * (r * r' - s * s')

    λ, V = eigen(Symmetric(ρ_new))
    λ_reg = map(e -> e < ρ_min ? ρ_min : (e > ρ_max ? ρ_max : e), λ)
    return V * diagm(λ_reg) * V'
end

function update_slack_and_dual_variable!(projection, constraint, primal, adaptive)
    @unpack ρ, z, α, r, s = constraint

    s .= -z

    z .= projection(primal + α)
    r .= primal - z
    α .+= r

    s .+= z
    s .= -ρ * s

    if adaptive == :mul
        ρ_new = mul_update_penalty_parameter(ρ, r, s)
    elseif adaptive == :add
        ρ_new = add_update_penalty_parameter(ρ, r, s)
    else
        return nothing
    end

    α .= inv(ρ_new) * ρ * α
    ρ .= ρ_new

    return nothing
end
