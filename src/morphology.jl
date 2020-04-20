using Statistics
using ImageTransformations: imrotate, center, Linear

###############################################################################
# Stacking routines

"""
    collapse(cube, [angles]; method=median, deweight=true, fill=0)

Combine all the frames of a cube using `method`. If `angles` are provided, will use [`derotate`](@ref) before combining.

If `deweight` is true, the method of Bottom et al. 2017 will be used in which the combined image will be the derotated weighted sum of the frames weighted by the temporal variance. `fill` will be passed to [`derotate`](@ref).

# Examples
```jldoctest
julia> X = ones(2, 3, 3)
2×3×3 Array{Float64,3}:
[:, :, 1] =
 1.0  1.0  1.0
 1.0  1.0  1.0

[:, :, 2] =
 1.0  1.0  1.0
 1.0  1.0  1.0

[:, :, 3] =
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> collapse(X)
3×3 Array{Float64,2}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> collapse(X, [0, 90], fill=NaN)
3×3 Array{Float64,2}:
 NaN    1.0  NaN
   1.0  1.0    1.0
 NaN    1.0  NaN

```

# See Also
[`collapse!`](@ref)
"""
function collapse(cube::AbstractArray{T,3}; method=median) where T
    return method(cube, dims = 1)[1, :, :]
end

function collapse(cube::AbstractArray{T,3}, angles::AbstractVector; method=median, deweight=true, fill=0) where T
    return deweight ? _collapse_deweighted(cube, angles, fill=fill) :
                      collapse(derotate(cube, angles, fill=fill); method = method)
end

"""
    collapse!(cube, angles; method=median, deweight=true)

An in-place version of the derotating [`collapse`](@ref). The only difference is in this version the cube will be derotated in-place.
"""
function collapse!(cube::AbstractArray{T,3}, angles::AbstractVector; method=median, deweight=true, fill=0) where T
    return deweight ? _collapse_deweighted!(cube, angles, fill=fill) :
                      collapse(derotate!(cube, angles, fill=fill); method = method)
end

 _collapse_deweighted(cube::AbstractArray{T,3}, angles::AbstractVector; fill=0) where T = 
    _collapse_deweighted!(deepcopy(cube), angles; fill=fill)
# deweight using Bottom et al. 2017
function _collapse_deweighted!(cube::AbstractArray{T,3}, angles::AbstractVector; fill=0) where T
    varframe = var(cube, dims=1)

    # have to check if no variance otherwise the returns will be NaN
    all(varframe .≈ 0) && return mean(derotate!(cube, angles; fill=fill); dims=1)[1, :, :]
    # create a cube from the variance of each pixel across time
    varcube = similar(cube)
    for idx in axes(vc, 1)
        @inbounds varcube[idx, :, :] .= varframe[1, :, :]
    end
    # derotate both cubes and perform a weighted sum
    derotate!(cube, angles; fill=fill)
    derotate!(varcube, angles; fill=fill)
    return sum(cube ./ varcube, dims=1)[1, :, :] ./ sum(inv.(varcube), dims=1)[1, :, :]
end

"""
    matrix(cube)

Given a cube of size `(n, x, y)` returns a matrix with size `(n, x * y)` where each row is a flattened image from the cube.

# Examples
```jldoctest
julia> X = ones(3, 2, 2)
3×2×2 Array{Float64,3}:
[:, :, 1] =
 1.0  1.0
 1.0  1.0
 1.0  1.0

[:, :, 2] =
 1.0  1.0
 1.0  1.0
 1.0  1.0

julia> matrix(X)
3×4 Array{Float64,2}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
```

# See Also
[`cube`](@ref)
"""
matrix(cube::AbstractArray{T,3}) where T = reshape(cube, size(cube, 1), size(cube, 2) * size(cube, 3))

"""
    cube(matrix)

Given a matrix of size `(n, z)`, returns a cube of size `(n, x, x)` where `x=√z`.

Will throw an error if `z` is not a perfect square.

# Examples
```jldoctest
julia> X = ones(3, 4)
3×4 Array{Float64,2}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0

julia> cube(X)
3×2×2 Array{Float64,3}:
[:, :, 1] =
 1.0  1.0
 1.0  1.0
 1.0  1.0

[:, :, 2] =
 1.0  1.0
 1.0  1.0
 1.0  1.0
```

# See Also
[`matrix`](@ref)
"""
function cube(mat::AbstractMatrix)
    n, z = size(mat)
    x = sqrt(z)
    isinteger(x) || error("Array of size $((n, x, x)) is not compatible with input matrix of size $(size(mat)).")
    return reshape(mat, n, Int(x), Int(x))
end


###############################################################################
# Rotation routines

"""
    derotate!(cube, angles; fill=0)

In-place version of [`derotate`](@ref)
"""
function derotate!(cube::AbstractArray{T,3}, angles::AbstractVector; fill=0) where T
    @inbounds for i in axes(cube, 1)
        frame = @view cube[i, :, :]
        cube[i, :, :] .= imrotate(frame, deg2rad(angles[i]), axes(frame), Linear(), fill)
    end
    return cube
end


"""
    derotate(cube, angles; fill=0)

Rotates an array using the given angles in degrees.

This will rotate frame `i` counter-clockwise by the amount `deg2rad(angles[i])`. Any values outside the original axes will be replaced with `fill`

# See Also
[`derotate!`](@ref), [`rotate`](@ref), [`rotate!`](@ref)
"""
function derotate(cube::AbstractArray{T,3}, angles::AbstractVector; fill=0) where T
    all(angles .≈ 0) && return cube
    return derotate!(deepcopy(cube), angles, fill=fill)
end
