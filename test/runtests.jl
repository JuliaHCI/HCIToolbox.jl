using HCIToolbox
using StableRNGs
using Statistics
using Test

rng = StableRNG(8799)

@testset "HCIToolbox.jl" begin
    include("morphology.jl")
    include("masking.jl")
    include("angles.jl")
    include("inject.jl")
    include("scaling.jl")
    include("geometry.jl")
    include("profiles.jl")
end
