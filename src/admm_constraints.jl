# ADMM constraints
struct ADMMConstraint{T}
    ρ::Vector{T} # penalty parameter
    z::Vector{T} # slack variable
    λ::Vector{T} # dual variable

    r::Vector{T} # primal residual
    s::Vector{T} # dual residual

    function ADMMConstraint{T}(n) where {T}
        ρ = ones(T, n)
        z = zeros(T, n)
        λ = zeros(T, n)
        r = zeros(T, n)
        s = zeros(T, n)
        return new(ρ, z, λ, r, s)
    end
end

mutable struct Constraints{T}
    terminal_state_projection::Union{Function,Nothing}
    terminal_state_constraint::ADMMConstraint{T}

    step_projection::Union{Function,Nothing}
    step_constraint::Vector{ADMMConstraint{T}}

    function Constraints{T}(nx, nu, N) where {T}
        terminal_state_projection = nothing
        terminal_state_constraint = ADMMConstraint{T}(nx)

        step_projection = nothing
        step_constraint = [ADMMConstraint{T}(nx+nu) for _ in 1:N]

        return new(terminal_state_projection, terminal_state_constraint, step_projection, step_constraint)
    end
end

## projection setting functions
function set_terminal_state_projection_function!(workset, Π::Function)
    setproperty!(workset.constraints, :terminal_state_projection, Π)
end

function set_step_projection_function!(workset, Π::Function)
    @unpack nx, nu = workset
    stacked_projection(arg) = Π(view(arg, 1:nx), view(arg, nx+1:nx+nu))
    setproperty!(workset.constraints, :step_projection, stacked_projection)
end

## penalty parameter setting functions
function set_constraint_parameter!(constraint, ρ_new)
    @unpack λ, ρ = constraint
    λ .*= ρ ./ ρ_new
    λ .= map(λ_k -> isnan(λ_k) ? 0 : λ_k, λ)
    ρ .= ρ_new
    return nothing
end

function set_terminal_state_constraint_parameter!(workset, ρ_new)
    set_constraint_parameter!(workset.constraints.terminal_state_constraint, ρ_new)
    return nothing
end

function set_step_constraint_parameter!(workset, ρ_new)
    ThreadsX.map(constraint -> set_constraint_parameter!(constraint, ρ_new), workset.constraints.step_constraint)
    return nothing
end

## evaluation functions
function add_penalty_derivative!(gradient, hessian, constraint, primal)
    @unpack ρ, z, λ = constraint
    gradient .+= (primal - z + λ) .* ρ
    hessian[diagind(hessian)] .+= ρ
    return nothing
end

function evaluate_penalty(constraint, primal)
    @unpack ρ, z, λ = constraint
    mapreduce((a, p) -> p / 2 * a^2, +, primal - z + λ, ρ)
end

function mul_update_penalty_parameter(ρ, r, s, η=0.1, ϵ=1e-12, ρ_max=1e8, ρ_min=1e-8)
    r = max(abs(r), ϵ)
    s = max(abs(s), ϵ)
    ρ *= exp(η * log(r^2 / s^2))

    # r2 = max(r^2, ϵ)
    # s2 = max(s^2, ϵ)
    # ρ *= exp(η * log(r2 / s2))

    return ρ <= ρ_min ? ρ_min : (ρ >= ρ_max ? ρ_max : ρ)
end

function add_update_penalty_parameter(ρ, r, s, λ=0.1, ρ_max=1e8, ρ_min=1e-8)
    ρ += λ * (r^2 - s^2)
    return ρ <= ρ_min ? ρ_min : (ρ >= ρ_max ? ρ_max : ρ)
end

function update_slack_and_dual_variable!(projection, constraint, primal, adaptive)
    @unpack ρ, z, λ, r, s = constraint

    s .= -z

    z .= projection(primal + λ)
    r .= primal - z
    λ .+= r

    s .+= z
    s .*= -ρ

    if adaptive == :mul
        ρ_new = mul_update_penalty_parameter.(ρ, r, s)
    elseif adaptive == :add
        ρ_new = add_update_penalty_parameter.(ρ, r, s)
    else
        return nothing
    end

    λ .*= ρ ./ ρ_new
    ρ .= ρ_new

    return nothing
end
