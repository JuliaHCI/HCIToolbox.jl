using HCIToolbox
using Test
using Statistics
using Random
Random.seed!(8799)

@testset "HCIToolbox.jl" begin
    include("stacking.jl")
    include("decomposition.jl")
end
