using Statistics
using Photometry.Aperture
using Rotations
using CoordinateTransformations
using ImageTransformations
using Interpolations
using StaticArrays

using PSFModels: PSFModel

export normalize_psf, normalize_psf!


struct CubeGenerator{CT<:AbstractArray,PT<:Union{AbstractExtrapolation,PSFModel},AT<:AbstractVector}
    cube::CT
    angles::AT
    psf::PT
end

function CubeGenerator(cube, angles, psf; kwargs...)
    psf′ = prep_psf(psf; kwargs...)
    return CubeGenerator(cube, angles, psf′)
end

function prep_psf(psf::AbstractMatrix{T}; degree=Linear(), fill=zero(T)) where T
    return ImageTransformations.box_extrapolation(psf, degree, fill)
end
prep_psf(psf::PSFModel; kwargs...) = psf

(gen::CubeGenerator)(pos; kwargs...) = gen(eltype(gen.cube), pos; kwargs...)

function (gen::CubeGenerator)(T::Type, pos; kwargs...)
    base = similar(gen.cube, T)
    gen(base, pos; kwargs...)
end

function (gen::CubeGenerator)(base::AbstractArray{T,3}, pos; A=one(T)) where T
    ctr = reverse(center(base)[2:3])
    xy = parse_position(pos, ctr)
    Threads.@threads for tidx in axes(base, 1)
        # position angle is 90° out of phase with parallactic angle
        angle = 90 - gen.angles[tidx]
        ϕ = RotMatrix{2}(deg2rad(angle))
        # reverse location to get in indices [y, x]
        location = recenter(ϕ, ctr)(xy) |> reverse

        frame = @view base[tidx, :, :]
        tform = Translation(center(gen.psf) - location)
        tform = ImageTransformations.try_static(tform, frame)
        for idx in CartesianIndices(frame)
            pidx = tform(SVector(idx.I))
            frame[idx] = A * gen.psf(Tuple(pidx)...)
        end
    end
    return base
end

function (gen::CubeGenerator)(base::AbstractMatrix{T}, pos; A=one(T)) where T
    ctr = reverse(center(gen.cube)[2:3])
    xy = parse_position(pos, ctr)
    Threads.@threads for tidx in axes(base, 1)
        # position angle is 90° out of phase with parallactic angle
        angle = 90 - gen.angles[tidx]
        ϕ = RotMatrix{2}(deg2rad(angle))
        # reverse location to get in indices [y, x]
        location = recenter(ϕ, ctr)(xy) |> reverse

        tform = Translation(center(gen.psf) - location)
        for (pidx, pidx′) in zip(axes(base, 2), CartesianIndices(size(gen.cube)[2:3]))
            I = tform(SVector(pidx′.I))
            base[tidx, pidx] = A * gen.psf(Tuple(I)...)
        end
    end
    return base
end

function (gen::CubeGenerator{<:AnnulusView})(base::AbstractArray{T,3}, pos; A=one(T)) where T
    dims = map(length, gen.cube.indices)
    mat = similar(gen.cube, T, dims...)
    gen(T, mat, pos; A=A)
    return inverse!(gen.cube, base, mat)
end


function (gen::CubeGenerator{<:AnnulusView})(base::AbstractMatrix{T}, pos; A=one(T)) where T
    ctr = reverse(center(gen.cube)[2:3])
    xy = parse_position(pos, ctr)
    Threads.@threads for tidx in axes(base, 1)
        # get tidx for view
        tidx′ = gen.cube.indices[1][tidx]
        # position angle is 90° out of phase with parallactic angle
        angle = 90 - gen.angles[tidx′]
        ϕ = RotMatrix{2}(deg2rad(angle))
        # reverse location to get in indices [y, x]
        location = recenter(ϕ, ctr)(xy) |> reverse
        tform = Translation(center(gen.psf) - location)
        for (pidx, pidx′) in zip(axes(base, 2), gen.cube.indices[2])
            I = tform(SVector(pidx′.I))
            base[tidx, pidx] = A * gen.psf(Tuple(I)...)
        end
    end

    return base
end


parse_position(pos::Tuple, ctr) = SVector(pos)
parse_position(pos::Polar, ctr) = convert(SVector, pos) |> Translation(ctr)


function _frame_center(axes)
    map(axes) do axis
        l, u = extrema(axis)
        (u - l) / 2 + 1
    end |> reverse |> SVector
end

_frame_center(img::AbstractMatrix) = ImageTransformations.center(img) |> reverse

# cartesian
function _get_location(ctr, pars::NamedTuple{(:x, :y)}, pa=0)
    pos = SVector(pars.x, pars.y)
    return recenter(RotMatrix{2}(deg2rad(-pa)), ctr)(pos)
end

_get_location(ctr, pars::NamedTuple{(:y, :x)}, pa=0) = _get_location(ctr, (x=pars.x, y=pars.y), pa)

# polar
function _get_location(ctr, pars::NamedTuple{(:r, :theta)}, pa=0)
    pos = Polar(promote(pars.r, deg2rad(pars.theta - pa))...)
    return CartesianFromPolar()(pos) + ctr
end

_get_location(ctr, pars::NamedTuple{(:theta, :r)}, pa=0) = _get_location(ctr, (r=pars.r, theta=pars.theta), pa)
_get_location(ctr, pars::NamedTuple{(:r, :θ)}, pa=0) = _get_location(ctr, (r=pars.r, theta=pars.θ), pa)
_get_location(ctr, pars::NamedTuple{(:θ, :r)}, pa=0) = _get_location(ctr, (r=pars.r, theta=pars.θ), pa)

# default is just the center
_get_location(ctr, pars::NamedTuple{()}, pa=0) = ctr


################################

"""
    inject(frame, psf; A=1, degree=Linear(), location...)

Injects the given PSF kernel or image into `frame` with amplitude `A`. The location can be given in image or polar coordinates, following the coordinate convention below. If no `location` is given, will assume the center of the frame. For empirical PSFs, `degree` is the corresponding `Interpolations.Degree` for the B-Spline used to subsample the pixel values.

### Coordinate System
* `x, y` - Parsed as distance from the bottom-left corner of the image. Pixel convention is that `(1, 1)` is the center of the bottom-left pixel increasing right and up.
* `r, theta` or `r, θ` - Parsed as polar coordinates from the center of the image. `theta` is a position angle.

!!! note
    Due to the integral nature of array indices, frames or images with even-sized axes will have their center rounded to the nearest integer. This may cause unexpected results for small frames and images.

# Examples
```jldoctest
julia> inject(zeros(5, 5), ones(1, 1), A=2, x=2, y=1) # image coordinates
5×5 Array{Float64,2}:
 0.0  2.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> inject(zeros(5, 5), ones(3, 3), r=1.5, theta=90) # polar coords
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  0.0
```
"""
inject(frame::AbstractMatrix, kernel; kwargs...) = inject!(copy(frame), kernel; kwargs...)

function inject(cube::AbstractArray{T,3}, kernel::AbstractMatrix, angles::AbstractVector; degree=Linear(), A=1, location...) where T
    size(cube, 1) == length(angles) ||
        error("Number of ADI frames does not much between cube ($(size(cube, 1))) and angles ($(length(angles)))")

    etp = ImageTransformations.box_extrapolation(kernel, degree, zero(T))

    ctr = center(cube)[2:3]
    idx = firstindex(cube, 1)
    frames = mapslices(cube, dims=[2,3]) do frame
        # frame = @view cube[idx, :, :]
        # have to do reversing and flip sign because `warp` moves canvas not image
        pos = _get_location(ctr, values(location), angles[idx]) |> reverse
        idx += 1
        tform = Translation(center(kernel) - pos)
        scaledwarp(frame, etp, ImageTransformations.try_static(tform, frame), A)
    end
    return frames
end


"""
    inject!(frame, ::PSFKernel; A=1, location...)
    inject!(frame, ::AbstractMatrix; A=1, degree=Linear(), location...)

In-place version of [`inject`](@ref) which modifies `frame`.
"""
function inject!(frame::AbstractMatrix{T}, kernel; A=1, pa=0, location...) where T
    ctr = _frame_center(frame)
    x0, y0 = _get_location(ctr, values(location), pa)
    for idx in CartesianIndices(frame)
        d = sqrt((idx.I[1] - y0)^2 + (idx.I[2] - x0)^2)
        frame[idx] += A * kernel(d)
    end

    return frame
end

function inject!(frame::AbstractMatrix{T}, kernel::AbstractMatrix; kwargs...) where T
    return frame .+= construct(kernel, axes(frame); kwargs...)
end


"""
    inject(cube, psf, [angles]; A=1, degree=Linear(), location...)

Injects `A * img` into each frame of `cube` at the position given by the keyword arguments. If `angles` are provided, the position in the keyword arguments will correspond to the `img` position on the first frame of the cube, with each subsequent repositioned `img` being rotated by `-angles` in degrees. This is useful for fake companion injection. If no `location` is given, will assume the center of each frame. For empirical PSFs, `degree` is the corresponding `Interpolations.Degree` for the B-Spline used to subsample the pixel values.
"""
inject(cube::AbstractArray{T,3}, args...; kwargs...) where T =
    inject!(copy(cube), args...; kwargs...)

"""
    inject!(cube, ::PSFKernel, [angles]; A=1, location...)
    inject!(cube, ::AbstractMatrix, [angles]; A=1, location...)

In-place version of [`inject`](@ref) which modifies `cube`.
"""
function inject!(cube::AbstractArray{T,3}, kernel; kwargs...) where T
    Threads.@threads for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        inject!(frame, kernel; kwargs...)
    end
    return cube
end

function inject!(cube::AbstractArray{T,3}, kernel, angles::AbstractVector; kwargs...) where T
    size(cube, 1) == length(angles) ||
        error("Number of ADI frames does not much between cube ($(size(cube, 1))) and angles ($(length(angles)))")
    Threads.@threads for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        inject!(frame, kernel; pa=angles[idx], kwargs...)
    end
    return cube
end

function inject!(cube::AbstractArray{T,3}, kernel::AbstractMatrix; degree=Linear(), A=1, location...) where T
    etp = ImageTransformations.box_extrapolation(kernel, degree, zero(T))

    ctr = center(cube)[2:3]

    Threads.@threads for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        # have to do reversing and flip sign because `warp` moves canvas not image
        pos = _get_location(ctr, values(location)) |> reverse
        tform = Translation(center(kernel) - pos)
        scaledwarp!(frame, etp, ImageTransformations.try_static(tform, frame), A)
    end
    return cube
end

function inject!(cube::AbstractArray{T,3}, kernel::AbstractMatrix, angles::AbstractVector; degree=Linear(), A=1, location...) where T
    size(cube, 1) == length(angles) ||
        error("Number of ADI frames does not much between cube ($(size(cube, 1))) and angles ($(length(angles)))")

    etp = ImageTransformations.box_extrapolation(kernel, degree, zero(T))

    ctr = center(cube)[2:3]

    Threads.@threads for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        # have to do reversing and flip sign because `warp` moves canvas not image
        pos = _get_location(ctr, values(location), angles[idx]) |> reverse
        tform = Translation(center(kernel) - pos)
        scaledwarp!(frame, etp, ImageTransformations.try_static(tform, frame), A)
    end
    return cube
end

function scaledwarp(frame, etp, tform, A)
    map(CartesianIndices(frame)) do I
        frame[I] + A * etp(Tuple(tform(SVector(I.I)))...)
    end
end

function scaledwarp!(frame, etp, tform, A)
    @inbounds for I in CartesianIndices(frame)
        frame[I] += A * etp(Tuple(tform(SVector(I.I)))...)
    end
    frame
end

###################

function normalize_psf!(psf::AbstractMatrix, fwhm, factor=1)
    ap = CircularAperture(center(psf), factor * fwhm / 2)
    area = photometry(ap, psf).aperture_sum
    return psf ./= area
end
normalize_psf(psf, fwhm, factor=1) = normalize_psf!(deepcopy(psf), fwhm, factor)
