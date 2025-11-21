using Documenter, IterativeLQR
using Literate

SOURCE = joinpath(@__DIR__, "..", "examples", "cartpole", "swing_up.jl")
OUTPUT = joinpath(@__DIR__, "src", "generated")

Literate.markdown(SOURCE, OUTPUT, codefence="```julia" => "```")
Literate.script(SOURCE, OUTPUT)

makedocs(
    sitename="IterativeLQR.jl",
    pages=[
        "Home" => "index.md",
        "Getting started" => "guide.md",
        "Examples" => [
            "Cartpole swing-up" => "generated/swing_up.md"
        ],
        "reference.md"
    ],
    repo = Remotes.GitHub("JurajLieskovsky", "IterativeLQR.jl")
    # repo="github.com/JurajLieskovsky/IterativeLQR.jl.git"
)


deploydocs(
    repo = Remotes.GitHub("JurajLieskovsky", "IterativeLQR.jl")
    # repo="github.com/JurajLieskovsky/IterativeLQR.jl.git"
)
