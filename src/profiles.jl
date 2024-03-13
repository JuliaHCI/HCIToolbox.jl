using ImageTransformations: center
using StatsBase
using NaNMath

"""
    radial_profile(image; center=center(image))

Calculates the radial profile of `image`, centered around `center`. Returns the radii and corresponding profile.

!!! note "Understanding bin edges"

    The radii returned here are integral, for example `[0, 1, 2, ...]`. These are equivalent to the centers of the contours of an annulus between `r - 0.5, r + 0.5` pixels. These annuli are centered on pixels (at least those orthogonal to the axes). 

# Examples

```jldoctest
julia> X = [0 1 1 1 0
            1 2 2 2 1
            1 2 3 2 1
            1 2 2 2 1
            0 1 1 1 0];

julia> r, prof = radial_profile(X);

julia> r .=> prof # radius => value
4-element Vector{Pair{Int64, Float64}}:
 0 => 3.0
 1 => 2.0
 2 => 1.0
 3 => 0.0

julia> radial_profile([1 2; 2 2], center=(1, 1))
([0, 1], [1.0, 2.0])
```
"""
function radial_profile(image::AbstractMatrix; center=center(image))
    inds = CartesianIndices(image)
    distance = vec(map(idx -> round(Int, sqrt((idx.I[1] - center[1])^2 + (idx.I[2] - center[2])^2)), inds))
    weights = vec(image)
    
    binned = sort(countmap(distance, weights))
    nr = sort(countmap(distance))
    
    radii = collect(keys(binned))
    profile = values(binned) ./ values(nr)
    return radii, profile
end

robust_radial_profile(image::AbstractMatrix; kwargs...) = robust_radial_profile(NaNMath.median, image; kwargs...)

function robust_radial_profile(func, image::AbstractMatrix; center=center(image), width=1)
    inds = CartesianIndices(image)
    distance = map(idx -> round(Int, sqrt((idx.I[1] - center[1])^2 + (idx.I[2] - center[2])^2)), inds)
    weights = image

    min_dist, max_dist = extrema(distance)
    bins = collect(min_dist:width:max_dist)
    vals = map(bins) do bin
        mask = bin .<= distance .< bin + width
        any(mask) || return NaN
        vals = weights[distance .== bin]
        return func(vals)
    end
    return bins, vals
end
