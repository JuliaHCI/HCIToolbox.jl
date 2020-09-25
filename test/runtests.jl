using HCIToolbox
using Test
using Statistics
using Random
Random.seed!(8799)

@testset "HCIToolbox.jl" begin
    include("morphology.jl")
    include("masking.jl")
    include("angles.jl")
    include("inject.jl")
    include("scaling.jl")
end
