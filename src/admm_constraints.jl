# ADMM constraints

abstract type Constraint end

mutable struct SingleConstraint{T} <: Constraint
    param::T                             # penalty parameter   
    projection::Union{Function,Nothing}  # projection function

    function SingleConstraint{T}(n) where {T}
        return new{T}(one(T), nothing)
    end
end

mutable struct MultipleConstraint{T} <: Constraint
    param::T                             # penalty parameter   
    projection::Union{Function,Nothing}  # projection function

    function MultipleConstraint{T}(n, m) where {T}
        return new{T}(one(T), nothing)
    end
end

isactive(constraint::Constraint) = constraint.projection !== nothing

## workset functions
function set_projection_function!(workset, constraint::Symbol, projection::Function)
    setproperty!(getproperty(workset, constraint), :projection, projection)
end

function set_terminal_state_constraint_parameter!(workset, ρ)
    workset.α .*= workset.terminal_state_constraint.param / ρ
    workset.terminal_state_constraint.param = ρ
    return nothing
end

function set_input_constraint_parameter!(workset, ρ)
    ThreadsX.map(u -> u ./= workset.input_constraint.param / ρ, workset.β)
    workset.input_constraint.param = ρ
    return nothing
end

## evaluation functions
function add_penalty_derivative!(gradient, hessian, param, primal, slack, dual)
    gradient .+= param * (primal - slack + dual)
    hessian[diagind(hessian)] .+= param
end

evaluate_penalty(parameter, residual, dual) = parameter / 2 * mapreduce(a -> a^2, +, residual + dual)

function update_slack_and_dual_variable!(projection, primal, slack, dual)
    slack .= projection(primal + dual)
    dual .+= primal - slack
end
