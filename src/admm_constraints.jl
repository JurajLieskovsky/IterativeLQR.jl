# ADMM constraints
struct ADMMConstraint{T}
    ρ::Vector{T} # penalty parameter
    z::Vector{T} # slack variable
    α::Vector{T} # dual variable

    r::Vector{T} # primal residual
    s::Vector{T} # dual residual

    function ADMMConstraint{T}(n) where {T}
        ρ = ones(T, n)
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
    α .*= ρ ./ ρ_new
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
    gradient .+= (primal - z + α) .* ρ
    hessian[diagind(hessian)] .+= ρ
    return nothing
end

function evaluate_penalty(constraint, primal)
    @unpack ρ, z, α = constraint
    mapreduce((a, p) -> p / 2 * a^2, +, primal - z + α, ρ)
end

function update_penalty_parameter(ρ, r, s, rt=2, ϵ=1e-8, ρ_max=1e8, ρ_min=1e-8)
    ρ *= ((r^2 + ϵ) / (s^2 + ϵ))^(1 / rt)
    return ρ <= ρ_min ? ρ_min : (ρ >= ρ_max ? ρ_max : ρ)
end

# function update_penalty_parameter(ρ, r, s, α=0.1, ρ_max=1e8, ρ_min=1e-8)
#     ρ += α * (r^2 - s^2)
#     return ρ <= ρ_min ? ρ_min : (ρ >= ρ_max ? ρ_max : ρ)
# end

function update_slack_and_dual_variable!(projection, constraint, primal, adaptive)
    @unpack ρ, z, α, r, s = constraint

    s .= -z

    z .= projection(primal + α)
    r .= primal - z
    α .+= r

    s .+= z
    s .*= -ρ

    if adaptive
        ρ_new = update_penalty_parameter.(ρ, r, s)
        α .*= ρ ./ ρ_new
        ρ .= ρ_new
    end

    return nothing
end
