using Statistics
using ImageTransformations
using CoordinateTransformations
using Interpolations

###############################################################################
# Stacking routines

"""
    collapse(cube; method=median, fill=0, degree=Linear())
    collapse(cube, angles; method=:deweight, fill=0, degree=Linear())

Combine all the frames of a cube using `method`. If `angles` are provided, will use [`derotate`](@ref) before combining.

If `method` is `:deweight`, the method of _Bottom et al. 2017_ will be used in which the combined image will be the derotated weighted sum of the frames weighted by the temporal variance. Keyword arguments will be passed to [`derotate`](@ref).

### References
1. [Bottom et al. 2017 "Noise-weighted Angular Differential Imaging"](https://ui.adsabs.harvard.edu/abs/2017RNAAS...1...30B)

# Examples
```jldoctest
julia> X = ones(2, 3, 3);

julia> collapse(X)
3×3 Array{Float64,2}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> collapse(X, [0, 90])
3×3 Array{Float64,2}:
 0.5  1.0  0.5
 1.0  1.0  1.0
 0.5  1.0  0.5

julia> collapse(X, [0, 90], fill=NaN)
3×3 Array{Float64,2}:
 NaN    1.0  NaN
   1.0  1.0    1.0
 NaN    1.0  NaN

```

# See Also
[`collapse!`](@ref)
"""
collapse(cube::AbstractArray{T,3}; method=median) where {T} = method(cube, dims = 1)[1, :, :]

collapse(cube::AbstractArray{T,3}, angles::AbstractVector; method=:deweight, kwargs...) where T =
    method === :deweight ? _collapse_deweighted(cube, angles; kwargs...) :
               collapse(derotate(cube, angles; kwargs...); method = method)

"""
    collapse!(cube, angles; method=:deweight, fill=0)

An in-place version of the derotating [`collapse`](@ref). The only difference is in this version the cube will be derotated in-place.
"""
collapse!(cube::AbstractArray{T,3}, angles::AbstractVector; method=:deweight, kwargs...) where T =
    method === :deweight ? _collapse_deweighted!(cube, angles; kwargs...) :
               collapse(derotate!(cube, angles; kwargs...); method = method)

# deweight using Bottom et al. 2017
 _collapse_deweighted(cube::AbstractArray{T,3}, angles::AbstractVector; kwargs...) where T =
    _collapse_deweighted!(deepcopy(cube), angles; kwargs...)

function _collapse_deweighted!(cube::AbstractArray{T,3}, angles::AbstractVector; fill=zero(T), kwargs...) where T
    varframe = var(cube, dims=1)

    # have to check if no variance otherwise the returns will be NaN
    all(varframe .≈ 0) && return mean(derotate!(cube, angles; fill=fill, kwargs...); dims=1)[1, :, :]
    # create a cube from the variance of each pixel across time
    varcube = similar(cube)
    Threads.@threads for idx in axes(varcube, 1)
        @inbounds varcube[idx, :, :] .= varframe[1, :, :]
    end
    # derotate both cubes and perform a weighted sum
    derotate!(cube, angles; fill=fill, kwargs...)
    derotate!(varcube, angles; fill=fill, kwargs...)

    # calculate collapsed sum and replace NaNs with our fill value
    out = sum(cube ./ varcube, dims=1) ./ sum(inv.(varcube), dims=1)
    @. out[isnan(out)] = fill
    return out[1, :, :]
end

"""
    flatten(cube)

Given a cube of size `(n, x, y)` returns a matrix with size `(n, x * y)` where each row is a flattened image from the cube.

# Examples
```jldoctest
julia> X = ones(3, 2, 2);

julia> flatten(X)
3×4 Array{Float64,2}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
```

# See Also
[`expand`](@ref)
"""
flatten(cube::AbstractArray{T,3}) where T = reshape(cube, size(cube, 1), size(cube, 2) * size(cube, 3))
flatten(mat::AbstractMatrix) = mat

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

julia> expand(X)[1, :, :]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  1.0
```

# See Also
[`flatten`](@ref)
"""
function expand(mat::AbstractMatrix)
    n, z = size(mat)
    x = sqrt(z)
    isinteger(x) || error("Array of size $((n, x, x)) is not compatible with input matrix of size $(size(mat)).")
    return reshape(mat, n, Int(x), Int(x))
end
expand(cube::AbstractArray{T,3}) where {T} = cube


###############################################################################
# Rotation routines

"""
    derotate!(cube, angles; fill=0, degree=Linear())

In-place version of [`derotate`](@ref) which modifies `cube`.
"""
function derotate!(cube::AbstractArray{T,3},
                   angles::AbstractVector;
                   fill=zero(T),
                   degree=Linear()) where T
    Threads.@threads for i in axes(cube, 1)
        frame = @view cube[i, :, :]
        frame .= imrotate(frame, deg2rad(angles[i]), axes(frame), degree, fill)
    end
    return cube
end

"""
    derotate(frame, angle; fill=0, degree=Linear())

Rotates `frame` counter-clockwise by `angle`, given in degrees. This is merely a convenient wrapper around `ImageTransformations.imrotate`.
"""
function derotate(frame::AbstractMatrix{T},
                   angle;
                   fill=zero(T),
                   degree=Linear()) where T
    return imrotate(frame, deg2rad(angle), axes(frame), degree, fill)
end


"""
    derotate(cube, angles; fill=0, degree=Linear())

Rotates an array using the given angles in degrees.

This will rotate frame `i` counter-clockwise. Any values outside the original axes will be replaced with `fill`. If the given angles are true parallactic angles, the resultant cube will have each frame aligned with top pointing North. `degree` is the corresponding `Interpolations.Degree` for the B-Spline used to subsample the pixel values.

# Examples
```jldoctest
julia> X = zeros(1, 3, 3); X[1, 1, 2] = 1;

julia> X[1, :, :]
3×3 Array{Float64,2}:
 0.0  1.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> derotate(X, [90])[1, :, :]
3×3 Array{Float64,2}:
 0.0  3.22941e-16  0.0
 0.0  0.0          0.999991
 0.0  0.0          0.0
```

# See Also
[`derotate!`](@ref)
"""
function derotate(cube::AbstractArray{T,3}, angles::AbstractVector; kwargs...) where T
    all(iszero, angles) && return cube
    return derotate!(deepcopy(cube), angles; kwargs...)
end

#################################

"""
    shift_frame(frame, dx, dy; fill=0)
    shift_frame(frame, dpos; fill=0)

Shifts `frame` by `dx` and `dy` with bilinear interpolation. If necessary, empty indices will be filled with `fill`.

# Examples
```jldoctest
julia> shift_frame([0 0 0; 0 1 0; 0 0 0], 1, -1)
3×3 Array{Float64,2}:
 0.0  0.0  1.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> shift_frame(ans, (-1, 1), fill=NaN)
3×3 Array{Float64,2}:
 NaN    NaN    NaN
   0.0    1.0  NaN
   0.0    0.0  NaN
```
"""
function shift_frame(frame::AbstractMatrix{T}, dx, dy; fill=zero(T)) where T
    tform = Translation(-dy, -dx)
    return warp(frame, tform, axes(frame), fill)
end
shift_frame(frame::AbstractMatrix{T}, dpos; fill=zero(T)) where T = shift_frame(frame, dpos...; fill=fill)

"""
    shift_frame(cube, dx, dy; fill=0)
    shift_frame(cube, dpos; fill=0)

Shift each frame of `cube` by `dx` and `dy`, which can be integers or vectors. The change in position can be given as a tuple, which can also be put into a vector to use across the cube. If a frame is shifted outside its axes, the empty indices will be filled with `fill`.

# See Also
[`shift_frame!`](@ref)
"""
shift_frame(cube::AbstractArray{T, 3}, dx, dy; fill=zero(T)) where T = shift_frame!(deepcopy(cube), dx, dy; fill=fill)
shift_frame(cube::AbstractArray{T, 3}, dpos; fill=zero(T)) where T = shift_frame!(deepcopy(cube), dpos; fill=fill)

"""
    shift_frame!(cube, dx, dy; fill=0)
    shift_frame!(cube, dpos; fill=0)

In-place version of [`shift_frame`](@ref) which modifies `cube`.
"""
function shift_frame!(cube::AbstractArray{T, 3}, dx::Number, dy::Number; fill=zero(T)) where T
    @inbounds for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        tform = Translation(-dy, -dx)
        frame .= warp(frame, tform, axes(frame), fill)
    end
    return cube
end

shift_frame!(cube::AbstractArray{T, 3}, dpos::Tuple; fill=zero(T)) where T = shift_frame!(cube, dpos...; fill=fill)

function shift_frame!(cube::AbstractArray{T, 3}, dx::AbstractVector, dy::AbstractVector; fill=zero(T)) where T
    @inbounds for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        tform = Translation(-dy[idx], -dx[idx])
        frame .= warp(frame, tform, axes(frame), fill)
    end
    return cube
end

function shift_frame!(cube::AbstractArray{T, 3}, dpos::AbstractVector{<:Tuple}; fill=zero(T)) where T
    @inbounds for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        dy, dx = dpos[idx]
        tform = Translation(-dy, -dx)
        frame .= warp(frame, tform, axes(frame), fill)
    end
    return cube
end

#################################

"""
    crop(frame, size; center=center(frame), force=false)
    crop(cube, size; center=center(frame), force=false)

Crop a frame or cube to `size`. `size` can be a tuple or an integer, which will make a square crop. The indices will be relative to `center`. To avoid allocations, consider [`cropview`](@ref).
"""
crop(input, size; kwargs...) = collect(cropview(input, size; kwargs...))

cropview(input, size::Integer; kwargs...) = cropview(input, (size, size); kwargs...)

"""
    cropview(cube::AbstractArray{T, 3}, size; center=center(frame), force=false)

Crop a frame to `size`, returning a view of the frame.

# See Also
* [`crop`](@ref)
"""
function cropview(cube::AbstractArray{T, 3}, size::Tuple; center=center(cube)[[2, 3]], force=false) where T    
    frame_size = (Base.size(cube, 2), Base.size(cube, 3))
    out_size = force ? size : check_size(frame_size, size)
    wing = @. (out_size - 1) / 2
    _init = @. floor(Int, center - wing)
    _end = @. floor(Int, center + wing)
    return view(cube, :, _init[1]:_end[1], _init[2]:_end[2])
end

"""
    cropview(frame::AbstractMatrix, size; center=center(frame), force=false)

Crop a frame to `size`, returning a view of the frame.

# See Also
* [`crop`](@ref)
"""
function cropview(frame::AbstractMatrix, size::Tuple; center=center(frame), force=false)
    out_size = force ? size : check_size(Base.size(frame), size)
    wing = @. (out_size - 1) / 2
    _init = @. floor(Int, center - wing)
    _end = @. floor(Int, center + wing)
    return view(frame, _init[1]:_end[1], _init[2]:_end[2])
end

"""
    HCIToolbox.check_size(frame_size, crop_size)

Given two image shapes, will adjust the output size to make sure even amounts are clipped off each side.
"""
function check_size(frame_size, crop_size)
    out = @. ifelse(iseven(frame_size - crop_size), crop_size, crop_size + 1)
    out != crop_size && @info "adjusted size to $out to avoid odd-sized cropping"
    return out
end
