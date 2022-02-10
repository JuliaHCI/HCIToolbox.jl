using ImageTransformations: center
using StatsBase

function radial_profile(image, center=center(image))    
    inds = CartesianIndices(image)
    distance = vec(map(idx -> round(Int, sqrt((idx.I[1] - center[1])^2 + (idx.I[2] - center[2])^2)), inds))
    weights = vec(image)
    
    profile = sort(countmap(distance, weights))
    nr = sort(countmap(distance))
    
    return collect(keys(profile)), values(profile) ./ values(nr)
end
