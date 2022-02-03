using CoordinateTransformations
using StaticArrays
using Base: @propagate_inbounds

inside_annulus(rmin, rmax, center, idx...) = inside_annulus(rmin, rmax, center, SVector(idx[begin:begin+1]))
inside_annulus(rmin, rmax, center, idx::CartesianIndex{2}) = inside_annulus(rmin, rmax, center, SVector(idx.I))
function inside_annulus(rmin, rmax, center::AbstractVector, position::AbstractVector)
    Δ = center - position
    r = sqrt(sum(abs2, Δ))
    return rmin ≤ r ≤ rmax
end

include("annulus.jl")
include("multiannulus.jl")