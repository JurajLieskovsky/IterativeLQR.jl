# ADMM constraints
struct Constraint{T}
    ρ::Vector{T} # penalty parameter
    z::Vector{T} # slack variable
    α::Vector{T} # dual variable

    function Constraint{T}(n) where T
        ρ = ones(T, n)
        z = zeros(T, n)
        α = zeros(T, n)
        return new(ρ, z, α)
    end
end

mutable struct Constraints{T}
    terminal_state_projection::Union{Function,Nothing}
    terminal_state_constraint::Constraint{T}

    input_projection::Union{Function,Nothing}
    input_constraint::Vector{Constraint{T}}

    function Constraints{T}(nx, nu, N) where {T}
        terminal_state_projection = nothing
        terminal_state_constraint = Constraint{T}(nx)

        input_projection = nothing
        input_constraint = [Constraint{T}(nu) for _ in 1:N]

        return new(terminal_state_projection, terminal_state_constraint, input_projection, input_constraint)
    end
end

## projection setting functions
function set_terminal_state_projection_function!(workset, Π::Function)
    setproperty!(workset.constraints, :terminal_state_projection, Π)
end

function set_input_projection_function!(workset, Π::Function)
    setproperty!(workset.constraints, :input_projection, Π)
end

## penalty parameter setting functions
function set_constraint_parameter!(parameter, dual, new_parameter)
    dual .*= parameter ./ new_parameter
    parameter .= new_parameter
    return nothing
end

function set_terminal_state_constraint_parameter!(workset, ρ_new)
    @unpack α, ρ = workset.constraints.terminal_state_constraint
    set_constraint_parameter!(ρ, α, ρ_new)
    return nothing
end

function set_input_constraint_parameter!(workset, ρ_new)
    ThreadsX.map(c -> set_constraint_parameter!(c.ρ, c.α, ρ_new), workset.constraints.input_constraint)
    return nothing
end

## evaluation functions
function add_penalty_derivative!(gradient, hessian, primal, constraint)
    @unpack ρ, z, α = constraint
    gradient .+= (primal - z + α) .* ρ
    hessian[diagind(hessian)] .+= ρ
    return nothing
end

function evaluate_penalty(primal, constraint)
    @unpack ρ, z, α = constraint
    mapreduce((a, p) -> p / 2 * a^2, +, primal - z + α, ρ)
end

function update_slack_and_dual_variable!(projection, primal, constraint)
    @unpack z, α = constraint
    z .= projection(primal + α)
    α .+= primal - z
    return nothing
end
