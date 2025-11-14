using Documenter, IterativeLQR

makedocs(
    sitename="IterativeLQR.jl",
    pages=[
        "Home" => "index.md",
        "Getting started" => "guide.md",
        "reference.md"
    ],
    remotes=nothing
)
