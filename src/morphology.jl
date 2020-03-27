using Statistics

###############################################################################
# Stacking routines

"""
    combine(cube, [angles]; method=median)

Combine all the frames of a cube using `method`. If `angles` are provided, will use [`derotate!`](@ref) before combining.

Note that with `angles` this is slightly slower than doing `combine(derotate!(cube, angles))`, if you can overwrite the input.

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

julia> combine(X)
3×3 Array{Float64,2}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> combine(X, [0, 90])
1×3×3 Array{Float64,3}:
[:, :, 1] =
 1.0  1.0  1.0

[:, :, 2] =
 NaN  1.0  NaN

[:, :, 3] =
 1.0  1.0  1.0
```
"""
function combine(cube::AbstractArray{T,3}; method = median) where T
    return method(cube, dims = 1)[1, :, :]
end

combine(cube::AbstractArray{T,3}, angles::AbstractVector; method = median) where T = combine(derotate(cube, angles); method = method)

"""
    flatten(cube)

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

julia> flatten(X)
3×4 Array{Float64,2}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
```

# See Also
* [`expand`](@ref)
"""
flatten(cube::AbstractArray{T,3}) where T = reshape(cube, size(cube, 1), size(cube, 2) * size(cube, 3))

"""
    expand(matrix)

Given a matrix of size `(n, z)`, returns a cube of size `(n, x, x)` where `x=√z`.

Will throw an error if `z` is not a perfect square.

# Examples
```jldoctest
julia> X = ones(3, 4)
3×4 Array{Float64,2}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0

julia> expand(X)
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
* [`flatten`](@ref)
"""
function expand(mat::AbstractMatrix)
    n, z = size(mat)
    x = sqrt(z)
    isinteger(x) || error("Array of size $((n, x, x)) is not compatible with input matrix of size $(size(mat)).")
    return reshape(mat, n, Int(x), Int(x))
end


###############################################################################
# Rotation routines

using ImageTransformations: imrotate, center


"""
    derotate!(cube, angles)

In-place version of [`derotate`](@ref)
"""
function derotate!(cube::AbstractArray{T,3}, angles::AbstractVector) where {T <: AbstractFloat}
    @inbounds for i in axes(cube, 1)
        frame = @view cube[i, :, :]
        cube[i, :, :] .= imrotate(frame, deg2rad(angles[i]), axes(frame))
    end
    return cube
end


"""
    derotate(cube, angles)

Rotates an array using the given angles in degrees.

This will rotate frame `i` counter-clockwise by the amount `deg2rad(angles[i])`.

# See Also
* [`derotate!`](@ref), [`rotate`](@ref), [`rotate!`](@ref)
"""
function derotate(cube::AbstractArray{T,3}, angles::AbstractVector) where T
    all(angles .≈ 0) && return cube
    return derotate!(deepcopy(cube), angles)
end

"""
    rotate!(cube, angles)

In-place version of [`derotate`](@ref)
"""
function rotate!(cube::AbstractArray{T,3}, angles::AbstractVector) where {T <: AbstractFloat}
    @inbounds for i in axes(cube, 1)
        frame = @view cube[i, :, :]
        cube[i, :, :] .= imrotate(frame, -deg2rad(angles[i]), axes(frame))
    end
    return cube
end


"""
    rotate(cube, angles)

Rotates an array using the given angles in degrees.

This will rotate frame `i` clockwise by the amount `deg2rad(angles[i])`.

# See Also
* [`rotate!`](@ref), [`derotate`](@ref), [`derotate!`](@ref)
"""
function rotate(cube::AbstractArray{T,3}, angles::AbstractVector) where T
    all(angles .≈ 0) && return cube
    return rotate!(deepcopy(cube), angles)
end
