using Documenter, IterativeLQR

makedocs(
    sitename="IterativeLQR.jl",
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Getting started" => "guide.md",
            "Examples" => "examples.md"
        ],
        "reference.md"
    ],
    remotes=nothing
)
