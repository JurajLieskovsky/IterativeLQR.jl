using Documenter, IterativeLQR

makedocs(
    sitename="IterativeLQR.jl",
    pages=[
        "Home" => "index.md",
        "Getting started" => "guide.md",
        "reference.md"
    ],
    repo = "github.com/JurajLieskovsky/IterativeLQR.jl.git"
)


deploydocs(
    repo = "github.com/JurajLieskovsky/IterativeLQR.jl.git"
)
