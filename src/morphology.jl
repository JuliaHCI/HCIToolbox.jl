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
        varcube[idx, :, :] .= varframe[1, :, :]
    end
    # derotate both cubes and perform a weighted sum
    derotate!(cube, angles; fill=fill)
    derotate!(varcube, angles; fill=fill)

    # calculate collapsed sum and replace NaNs with our fill value
    out = sum(cube ./ varcube, dims=1) ./ sum(inv.(varcube), dims=1)
    @. out[isnan(out)] = fill
    return out[1, :, :]
end

"""
    tomatrix(cube)

Given a cube of size `(n, x, y)` returns a matrix with size `(n, x * y)` where each row is a flattened image from the cube.

# Examples
```jldoctest
julia> X = ones(3, 2, 2);

julia> tomatrix(X)
3×4 Array{Float64,2}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
```

# See Also
[`tocube`](@ref)
"""
tomatrix(cube::AbstractArray{T,3}) where T = reshape(cube, size(cube, 1), size(cube, 2) * size(cube, 3))

"""
    tocube(matrix)

Given a matrix of size `(n, z)`, returns a cube of size `(n, x, x)` where `x=√z`.

Will throw an error if `z` is not a perfect square.

# Examples
```jldoctest
julia> X = ones(3, 4)
3×4 Array{Float64,2}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0

julia> tocube(X)
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
[`tomatrix`](@ref)
"""
function tocube(mat::AbstractMatrix)
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

################################

"""
    inject_image(frame, img; x, y, A=1)
    inject_image(frame, img; r, theta, A=1)

Injects `A * img` into `frame` at the position relative to the center of `frame` given by the keyword arguments. If necessary, `img` will be bilinearly interpolated onto the new indices. When used for injecting a PSF, it is imperative the PSF is already centered and, preferrably, odd-sized. 

### Coordinate System

The positions are decided in this way:
* `x, y` - Parsed as distance from the bottom-left corner of the image. Pixel convention is that `(1, 1)` is the center of the bottom-left pixel increasing right and up. 
* `r, theta` - Parsed as polar coordinates from the center of the image. `theta` is a position angle.

!!! note
    Due to the integral nature of array indices, frames or images with even-sized axes will have their center rounded to the nearest integer. This may cause unexpected results for small frames and images.

# Examples
```jldoctest
julia> inject_image(zeros(5, 5), ones(1, 1), A=2, x=2, y=1) # image coordinates
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  2.0  0.0  0.0  0.0

julia> inject_image(zeros(5, 5), ones(3, 3), r=1.5, theta=90) # polar coords
5×5 Array{Float64,2}:
 0.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  0.5  0.5  0.5  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```
"""
inject_image(frame::AbstractMatrix, img::AbstractMatrix; A=1, parametrization...) = 
    inject_image!(deepcopy(frame), img; A=A, parametrization...)

"""
    inject_image!(frame, img; x, y, A=1)
    inject_image!(frame, img; r, theta, A=1)

In-place version of [`inject_image`](@ref) which modifies `frame`.
"""
function inject_image!(frame::AbstractMatrix{T}, img::AbstractMatrix; A=1, parametrization...) where T
    # get the correct translation depending on (x,y) vs (r, θ)
    tform = _get_translation((;parametrization...), frame, img)
    # shift image with zero padding and add to frame
    shifted_img = warp(img, tform, axes(frame), Linear(), zero(T))
    return @. frame += A * shifted_img
end

"""
    inject_image(cube, img, [angles]; x, y, A=1)
    inject_image(cube, img, [angles]; r, theta, A=1)

Injects `A * img` into each frame of `cube` at the position given by the keyword arguments. If `angles` are provided, the position in the keyword arguments will correspond to the `img` position on the first frame of the cube, with each subsequent repositioned `img` being rotated by `-angles` in degrees. This is useful for fake companion injection.
"""
inject_image(cube::AbstractArray{T,3}, img; A=1, parametrization...) where T =
    inject_image!(deepcopy(cube), img; A=A, parametrization...)
inject_image(cube::AbstractArray{T,3}, img, angles; A=1, parametrization...) where T =
    inject_image!(deepcopy(cube), img, angles; A=A, parametrization...)

"""
    inject_image!(cube, img, [angles]; x, y, A=1)
    inject_image!(cube, img, [angles]; r, theta, A=1)

In-place version of [`inject_image`](@ref) which modifies `cube`.
"""
function inject_image!(cube::AbstractArray{T,3}, img::AbstractMatrix; A=1, parametrization...) where T
    for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        inject_image!(frame, img; A=A, parametrization...)
    end
    return cube
end

function inject_image!(cube::AbstractArray{T,3}, img::AbstractMatrix, angles::AbstractVector; A=1, parametrization...) where T
    for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        # frame rotation
        rot = LinearMap(RotMatrix{2}(-deg2rad(angles[idx])))
        # get the correct translation depending on (x,y) vs (r, θ) WITH the extra rotation
        tform = rot ∘ _get_translation((;parametrization...), frame, img)
        # transform image with zero padding and add to frame
        shifted_img = warp(img, tform, axes(frame), Linear(), zero(T))
        @. frame += A * shifted_img
    end
    return cube
end

#=
These functions allow dispatching on keyword arguments to inject_image!

The NamedTuples should constand fold during compilation, incurring no 
function calls for dispatching. Note the discrepancy in image coordinates 
to cartesian coordinates in the Translation
=#
_get_translation(pars::NamedTuple{(:x, :y)}, frame, img) =  Translation(-pars.y, -pars.x) ∘ Translation(center(img))
_get_translation(pars::NamedTuple{(:y, :x)}, frame, img) = _get_translation((x=pars.x, y=pars.y), frame, img)

function _get_translation(pars::NamedTuple{(:r, :theta)}, frame, img)
    # convert to position angle in radians
    angle = -deg2rad(pars.theta + 90)
    new_center = Polar(promote(pars.r, angle)...)
    # the translation
    trans = new_center |> CartesianFromPolar() |> Translation
    return trans ∘ Translation(center(img) - center(frame))
end
_get_translation(pars::NamedTuple{(:theta, :r)}, frame, img) = _get_translation((r=pars[2], theta=pars[1]), frame, img)
