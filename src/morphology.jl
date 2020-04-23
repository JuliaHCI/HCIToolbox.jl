using Statistics
using ImageTransformations
using CoordinateTransformations
using Interpolations

###############################################################################
# Stacking routines

"""
    collapse(cube; method=median, fill=0)
    collapse(cube, angles; method=median, deweight=true, fill=0)

Combine all the frames of a cube using `method`. If `angles` are provided, will use [`derotate`](@ref) before combining.

If `deweight` is true, the method of _Bottom et al. 2017_ will be used in which the combined image will be the derotated weighted sum of the frames weighted by the temporal variance. `fill` will be passed to [`derotate`](@ref).

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

# deweight using Bottom et al. 2017
 _collapse_deweighted(cube::AbstractArray{T,3}, angles::AbstractVector; fill=0) where T = 
    _collapse_deweighted!(deepcopy(cube), angles; fill=fill)
function _collapse_deweighted!(cube::AbstractArray{T,3}, angles::AbstractVector; fill=0) where T
    varframe = var(cube, dims=1)

    # have to check if no variance otherwise the returns will be NaN
    all(varframe .≈ 0) && return mean(derotate!(cube, angles; fill=fill); dims=1)[1, :, :]
    # create a cube from the variance of each pixel across time
    varcube = similar(cube)
    for idx in axes(varcube, 1)
        frame = @view varcube[idx, :, :]
        for ij in eachindex(frame)
            frame[ij] = max(one(T), varframe[ij])
        end
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
julia> X = ones(3, 2, 2);

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

In-place version of [`derotate`](@ref) which modifies `cube`.
"""
function derotate!(cube::AbstractArray{T,3}, angles::AbstractVector; fill=0) where T
    for i in axes(cube, 1)
        frame = @view cube[i, :, :]
        frame .= imrotate(frame, deg2rad(-angles[i]), axes(frame), Linear(), fill)
    end
    return cube
end


"""
    derotate(cube, angles; fill=0)

Rotates an array using the given angles in degrees.

This will rotate frame `i` counter-clockwise. Any values outside the original axes will be replaced with `fill`. If the given angles are true parallactic angles, the resultant cube will have each frame aligned with top pointing North.

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
 0.0       0.0          0.0
 0.999974  0.0          0.0
 0.0       8.71942e-15  0.0
```

# See Also
[`derotate!`](@ref), [`rotate`](@ref), [`rotate!`](@ref)
"""
function derotate(cube::AbstractArray{T,3}, angles::AbstractVector; fill=0) where T
    all(angles .≈ 0) && return cube
    return derotate!(deepcopy(cube), angles, fill=fill)
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
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  1.0

julia> shift_frame(ans, (-1, 1), fill=NaN)
 3×3 Array{Float64,2}:
    0.0    0.0  NaN
    0.0    1.0  NaN
  NaN    NaN    NaN
```
"""
function shift_frame(frame::AbstractMatrix{T}, dx, dy; fill=zero(T)) where T
    ctr = center(frame)
    tform = Translation(dy, -dx)
    return warp(frame, tform, axes(frame), fill)
end
shift_frame(frame::AbstractMatrix{T}, dpos; fill=zero(T)) where T = shift_frame(frame, dpos...; fill=fill)

"""
    shift_frame(cube, dx, dy; fill=0)
    shift_frame(cube, dpos; fill=0)

Shift each frame of `cube` by `dx` and `dy`, which can be integers or vectors. The change in position can be given as some iterable, which can also be put into a vector to use across the cube. If a frame is shifted outside its axes, the empty indices will be filled with `fill`.

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
        cube[idx, :, :] .= shift_frame(cube[idx, :, :], dx, dy; fill=fill)
    end
end

shift_frame!(cube::AbstractArray{T, 3}, dpos::AbstractVector{<:Number}; fill=zero(T)) where T = shift_frame!(cube, dpos...; fill=fill)

function shift_frame!(cube::AbstractArray{T, 3}, dx::AbstractVector, dy::AbstractVector; fill=zero(T)) where T
    @inbounds for idx in axes(cube, 1)
        cube[idx, :, :] .= shift_frame(cube[idx, :, :], dx[idx], dy[idx]; fill=fill)
    end
end

function shift_frame!(cube::AbstractArray{T, 3}, dpos::AbstractVector{<:AbstractVector}; fill=zero(T)) where T
    @inbounds for idx in axes(cube, 1)
        dx, dy = dpos[idx]
        cube[idx, :, :] .= shift_frame(cube[idx, :, :], dx, dy; fill=fill)
    end
end

################################

"""
    inject_image(frame, img; x, y)
    inject_image(frame, img; r, theta)

Injects `img` into `frame` at the position relative to the center of `frame` given by the keyword arguments. If necessary, `img` will be bilinearly interpolated onto the new indices. When used for injecting a PSF, it is imperative the PSF is already centered and, preferrably, odd-sized. 

!!! note
    Due to the integral nature of array indices, frames or images with even-sized axes will have their center rounded to the nearest integer. This may cause unexpected results for small frames and images.

# Examples
```jldoctest
julia> inject_image(zeros(5, 5), ones(1, 1), x=2, y=1)
5×5 Array{Float64,2}:
 0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> inject_image(zeros(5, 5), ones(3, 3), r=1.5, theta=90)
5×5 Array{Float64,2}:
 0.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  0.5  0.5  0.5  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```
"""
inject_image(frame::AbstractMatrix, img::AbstractMatrix; parametrization...) = 
    inject_image!(deepcopy(frame), img; parametrization...)

"""
    inject_image!(frame, img; x, y)
    inject_image!(frame, img; r, theta)

In-place version of [`inject_image`](@ref) which modifies `frame`.
"""
function inject_image!(frame::AbstractMatrix{T}, img::AbstractMatrix; parametrization...) where T
    # get the correct translation depending on (x,y) vs (r, θ)
    tform = Translation(center(img) - center(frame)) ∘ _get_translation((;parametrization...))
    # shift image with zero padding and add to frame, use view to avoid allocation
    shifted_img = warpedview(img, tform, axes(frame), Linear(), zero(T))
    return frame .+= shifted_img
end

"""
    inject_image(cube, img, [angles]; x, y)
    inject_image(cube, img, [angles]; r, theta)

Injects `img` into each frame of `cube` at the position given by the keyword arguments. If `angles` are provided, the position in the keyword arguments will correspond to the `img` position on the first frame of the cube, with each subsequent frame being rotated by `-angles`. This is useful for fake companion injection.
"""
inject_image(cube::AbstractArray{T,3}, img; parametrization...) where T =
    inject_image!(deepcopy(cube), img; parametrization...)
inject_image(cube::AbstractArray{T,3}, img, angles; parametrization...) where T =
    inject_image!(deepcopy(cube), img, angles; parametrization...)

function inject_image!(cube::AbstractArray{T,3}, img::AbstractMatrix; parametrization...) where T
    for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        inject_image!(frame, img; parametrization...)
    end
    return cube
end

function inject_image!(cube::AbstractArray{T,3}, img::AbstractMatrix, angles::AbstractVector; parametrization...) where T
    for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        inject_image!(frame, img; parametrization...)
        frame .= imrotate(frame, deg2rad(angles[idx]), axes(frame), Linear(), zero(T))
    end
    return cube
end

#=
These functions allow dispatching on keyword arguments to inject_image!

The NamedTuples should constand fold during compilation, incurring no 
function calls for dispatching. Note the discrepancy in image coordinates 
to cartesian coordinates in the Translation
=#
_get_translation(pars::NamedTuple{(:x, :y)}) = Translation(pars.y, -pars.x)
_get_translation(pars::NamedTuple{(:y, :x)}) = Translation(pars.y, -pars.x)

function _get_translation(pars::NamedTuple{(:r, :theta)})
    # convert to position angle in radians
    angle = deg2rad(-pars.theta - 90)
    new_center = Polar(promote(pars.r, angle)...)
   return new_center |> CartesianFromPolar() |> Translation
end
_get_translation(pars::NamedTuple{(:theta, :r)}) = _get_translation((r=pars[2], theta=pars[1]))
