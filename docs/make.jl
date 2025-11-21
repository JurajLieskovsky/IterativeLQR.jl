using Documenter, IterativeLQR
using Literate

for example in ("cartpole/swing_up.jl", "quadrotor/hover_to_hover.jl")
    SOURCE = joinpath(@__DIR__, "..", "examples", example)
    OUTPUT = joinpath(@__DIR__, "src", "generated")
    Literate.markdown(SOURCE, OUTPUT, codefence="```julia" => "```")
end

makedocs(
    sitename="IterativeLQR.jl",
    pages=[
        "Home" => "index.md",
        "Getting started" => "guide.md",
        "Examples" => [
            "Cart-pole swing-up" => "generated/swing_up.md"
            "Quadrotor recovery" => "generated/hover_to_hover.md"
        ],
        "reference.md"
    ],
    repo = "github.com/JurajLieskovsky/IterativeLQR.jl.git"
)

deploydocs(
    repo = "github.com/JurajLieskovsky/IterativeLQR.jl.git"
)
