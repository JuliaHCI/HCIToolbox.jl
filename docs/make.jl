using HCIToolbox
using Documenter

setup = quote
    using HCIToolbox
    using Random
    Random.seed!(8799)
end

DocMeta.setdocmeta!(HCIToolbox, :DocTestSetup, setup; recursive = true)


makedocs(;
    modules = [HCIToolbox],
    authors = "Miles Lucas <mdlucas@hawaii.edu>",
    repo = "https://github.com/juliahci/HCIToolbox.jl/blob/{commit}{path}#L{line}",
    sitename = "HCIToolbox.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliahci.github.io/HCIToolbox.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API/Reference" => "api.md"
    ],
)

deploydocs(;
    repo = "github.com/JuliaHCI/HCIToolbox.jl",
    push_preview = true
)
