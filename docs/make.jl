using HCIToolbox
using Documenter

makedocs(;
    modules=[HCIToolbox],
    authors="Miles Lucas <mdlucas@hawaii.edu>",
    repo="https://github.com/mileslucas/HCIToolbox.jl/blob/{commit}{path}#L{line}",
    sitename="HCIToolbox.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mileslucas.github.io/HCIToolbox.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mileslucas/HCIToolbox.jl",
)
