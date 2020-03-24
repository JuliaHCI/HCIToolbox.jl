using HCIToolbox
using Test
using Statistics
using Random
Random.seed!(8799)

@testset "HCIToolbox.jl" begin
    include("stacking.jl")
    include("masking.jl")
    include("decomposition.jl")
end
