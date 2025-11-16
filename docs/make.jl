using Documenter, IterativeLQR

makedocs(
    sitename="IterativeLQR.jl",
    pages=[
        "Home" => "index.md",
        "Getting started" => "guide.md",
        "reference.md"
    ],
)

deploydocs(
    repo = "github.com/JurajLieskovsky/IterativeLQR.jl.git",
)
