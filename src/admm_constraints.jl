# ADMM constraints
mutable struct Constraint{T}
    ρ::T                        # penalty parameter   
    Π::Union{Function,Nothing}  # projection function

    function Constraint{T}() where {T}
        return new{T}(one(T), nothing)
    end
end

isactive(constraint::Constraint) = constraint.Π !== nothing

## workset functions
function set_projection_function!(workset, constraint::Symbol, projection::Function)
    setproperty!(getproperty(workset, constraint), :Π, projection)
end

function set_terminal_state_constraint_parameter!(workset, ρ_new)
    workset.α .*= workset.terminal_state_constraint.ρ / ρ_new
    workset.terminal_state_constraint.ρ = ρ_new
    return nothing
end

function set_input_constraint_parameter!(workset, ρ_new)
    ThreadsX.map(u -> u ./= workset.input_constraint.ρ / ρ_new, workset.β)
    workset.input_constraint.ρ = ρ_new
    return nothing
end

## evaluation functions
function add_penalty_derivative!(gradient, hessian, constraint, primal, slack, dual)
    isactive(constraint) || return nothing
    gradient .+= constraint.ρ * (primal - slack + dual)
    hessian[diagind(hessian)] .+= constraint.ρ
    return nothing
end

function evaluate_penalty(constraint, residual, dual)
    isactive(constraint) || return nothing
    constraint.ρ / 2 * mapreduce(a -> a^2, +, residual + dual)
end

function update_slack_and_dual_variable!(Π, primal, slack, dual)
    slack .= Π(primal + dual)
    dual .+= primal - slack
end
