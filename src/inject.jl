using Statistics
using Photometry.Aperture
using FillArrays
using Rotations
using CoordinateTransformations
using ImageTransformations
using Interpolations
using StaticArrays

using PSFModels: PSFModel

export normalize_psf, normalize_psf!

const PSFType = Union{AbstractExtrapolation,PSFModel}

struct CubeGenerator{CT<:AbstractArray,PT<:PSFType,AT<:AbstractVector}
    cube::CT
    angles::AT
    psf::PT
end

"""
    CubeGenerator(cube, angles, psf; degree=Linear(), fill=0)

Creates a generator which can rapidly create synthetic data using a PSF. The `cube` and `angles` are used to define the geometry of the synthetic data cube. If the `psf` is a dense matrix, an extrapolator will be created with the given `degree` and `fill` value.

# Examples

```jldoctest
julia> using HCIDatasets: BetaPictoris

julia> cube, angles, psf = BetaPictoris[:cube, :pa, :psf];

julia> gen = CubeGenerator(cube, angles, psf); # using empirical PSF

julia> using PSFModels: Gaussian

julia> gen2 = CubeGenerator(cube, angles, Gaussian(4.7)); # using PSFModel
```
"""
function CubeGenerator(cube, angles, psf; kwargs...)
    if size(cube, 1) != length(angles)
        error("Number of frames in cube does not match number of parallactic angles")
    end
    psf′ = prep_psf(psf; kwargs...)
    return CubeGenerator(cube, angles, psf′)
end

function prep_psf(psf::AbstractMatrix{T}; degree=Linear(), fill=zero(T)) where T
    return ImageTransformations.box_extrapolation(psf, degree, fill)
end
prep_psf(psf::PSFModel; kwargs...) = psf

"""
    (::CubeGenerator)([T], pos; A=1)
    (::CubeGenerator)(base, pos; A=1)

Use the cube generator to generate a synthetic cube. The eltype can be manually specified with `T`, or otherwise a base can be given. If a base is given, the synthetic cube will be *injected*, which will add it to the `base`. `base` can be a matrix, in which case the generator will automatically flatten the output without extra allocations.

In the case that the generator is build on top of a geometry like [`AnnulusView`](@ref), using a matrix `base` will use the *filtered* matrix in order to avoid calculating out of bounds indices.

# Examples

```jldoctest
julia> using HCIDatasets: BetaPictoris

julia> cube, angles, psf = BetaPictoris[:cube, :pa, :psf];

julia> gen = CubeGenerator(cube, angles, psf); # using empirical PSF

julia> s = gen((51, 51)); # cube with PSF at center

julia> size(s) == size(cube)
true

julia> flat_s = gen(zero(flatten(cube)), (51, 51));

julia> flat_s ≈ flatten(s)
true

julia> gen(Polar(10, deg2rad(45)); A=10); # inject using Polar coords
```
"""
(gen::CubeGenerator)(pos; kwargs...) = gen(eltype(gen.cube), pos; kwargs...)

function (gen::CubeGenerator)(T::Type, pos; kwargs...)
    base = fill!(similar(gen.cube, T), zero(T))
    gen(base, pos; kwargs...)
end

function (gen::CubeGenerator)(base::AbstractArray{T,3}, pos; A=one(T)) where T
    ctr = reverse(center(base)[2:3])
    Threads.@threads for tidx in axes(base, 1)
        location = parse_position(pos, ctr, gen.angles[tidx])

        frame = @view base[tidx, :, :]
        tform = Translation(center(gen.psf) - location)
        tform = ImageTransformations.try_static(tform, frame)
        for idx in CartesianIndices(frame)
            pidx = tform(SVector(idx.I))
            if gen.psf isa PSFModel
                frame[idx] += A * gen.psf(Tuple(reverse(pidx))...)
            else
                frame[idx] += A * gen.psf(Tuple(pidx)...)
            end
        end
    end
    return base
end

function (gen::CubeGenerator)(base::AbstractMatrix{T}, pos; A=one(T)) where T
    ctr = reverse(center(gen.cube)[2:3])
    ny, nx = size(gen.cube)[2:3]
    Threads.@threads for tidx in axes(base, 1)
        location = parse_position(pos, ctr, gen.angles[tidx])
        tform = Translation(center(gen.psf) - location)
        for pidx in axes(base, 2)
            j = (pidx - 1) % ny + 1
            k = (pidx - 1) ÷ nx + 1
            I = tform(SVector(j, k))
            if gen.psf isa PSFModel
                base[tidx, pidx] += A * gen.psf(Tuple(reverse(I))...)
            else
                base[tidx, pidx] += A * gen.psf(Tuple(I)...)
            end
        end
    end
    return base
end

function (gen::CubeGenerator{<:AnnulusView})(base::AbstractArray{T,3}, pos; A=one(T)) where T
    dims = map(length, gen.cube.indices)
    mat = fill!(similar(gen.cube, T, dims...), zero(T))
    gen(mat, pos; A=A)
    return inverse!(gen.cube, base, mat)
end


function (gen::CubeGenerator{<:AnnulusView})(base::AbstractMatrix{T}, pos; A=one(T)) where T
    ctr = reverse(center(gen.cube)[2:3])
    Threads.@threads for tidx in axes(base, 1)
        location = parse_position(pos, ctr, gen.angles[tidx′])
        tform = Translation(center(gen.psf) - location)
        for (pidx, pidx′) in zip(axes(base, 2), gen.cube.indices[2])
            I = tform(SVector(pidx′.I))
            if gen.psf isa PSFModel
                base[tidx, pidx] += A * gen.psf(Tuple(reverse(I))...)
            else
                base[tidx, pidx] += A * gen.psf(Tuple(I)...)
            end
        end
    end
    return base
end

"""
    HCIToolbox.parse_position(pos::Tuple, ctr, pa=0)

Parse the position as the cartesian `(x, y)` pixel coordinates. If a parallactic angle is given the position will be rotated `pa` degrees clockwise.
"""
function parse_position(pos::Tuple, ctr, pa=0)
    xy = SVector(pos)
    ϕ = RotMatrix{2}(-deg2rad(pa))
    return recenter(ϕ, ctr)(xy) |> reverse
end

"""
    HCIToolbox.parse_position(pos::Polar, ctr, pa=0)

Parse the position as the polar `(r, θ)` pixel coordinates centered at `ctr`. If a parallactic angle is given the position will be rotated `pa` degrees clockwise.
"""
function parse_position(pos::Polar, ctr, pa=0)
    new_pos = Polar(pos.r, pos.θ - deg2rad(pa))
    Δxy = convert(SVector, new_pos)
    # recenter and switch to index layout
    return  Δxy |> Translation(ctr) |> reverse
end

################################

"""
    inject(frame, psf, position; A=1, degree=Linear(), fill=0.0)

Injects the matrix `psf` into `frame` at the given position. If `position` is a Tuple, it will be parsed as pixel coordinates `(x, y)`. If `position` is a `Polar` (from CoordinateTransformations.jl) it will be parsed as polar coordinates centered on the frame center.

The amplitude of the injected frame can be changed with `A`, and in the case that `psf` is dense matrix, and extrapolator will be created with the given `degree` and `fill` value.

### Coordinate System
* Cartesian - Parsed as distance from the bottom-left corner of the image. Pixel convention is that `(1, 1)` is the center of the bottom-left pixel increasing right and up.
* Polar - Parsed as polar coordinates from the center of the image. `theta` is a position angle.

!!! note
    Due to the integral nature of array indices, frames or images with even-sized axes will have their center rounded to the nearest integer. This may cause unexpected results for small frames and images.

# Examples
```jldoctest
julia> inject(zeros(5, 5), ones(1, 1), (2, 1); A=2) # image coordinates (x=2, y=1)
5×5 Array{Float64,2}:
 0.0  2.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> inject(zeros(5, 5), ones(3, 3), Polar(1.5, deg2rad(90))) # polar coords
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  0.0
```
"""
inject(frame::AbstractMatrix, args...; kwargs...) = inject!(copy(frame), args...; kwargs...)

function inject!(frame::AbstractMatrix, kernel::AbstractMatrix{T}, pos; degree=Linear(), fill=zero(T), kwargs...) where T
    etp = ImageTransformations.box_extrapolation(kernel, degree, fill)
    return inject!(frame, etp, pos; kwargs...)
end

function inject!(frame::AbstractMatrix{T}, psf::PSFType, pos; A=one(T)) where T
    ctr = reverse(center(frame))
    location = parse_position(pos, ctr)
    tform = Translation(center(psf) - location)
    tform = ImageTransformations.try_static(tform, frame)
    for idx in CartesianIndices(frame)
        I = tform(SVector(idx.I))
        frame[idx] += A * psf(Tuple(I)...)
    end

    return frame
end

################################

"""
    inject(cube, [angles], psf, position; A=1, degree=Linear(), fill=0.0)


Injects the matrix `psf` into each frame of `cube` at the given position. If `position` is a Tuple, it will be parsed as pixel coordinates `(x, y)`. If `position` is a `Polar` (from CoordinateTransformations.jl) it will be parsed as polar coordinates centered on the frame center.

If `angles` are provided, each position will be rotated by the give parallactic angles in degrees.

The amplitude of the injected frame can be changed with `A`, and in the case that `psf` is dense matrix, an extrapolator will be created with the given `degree` and `fill` value.
"""
inject(cube::AbstractArray{T,3}, args...; kwargs...) where T =
    inject!(copy(cube), args...; kwargs...)

"""
    inject!(cube, [angles], psf, position; A=1, degree=Linear(), fill=0.0)

In-place version of [`inject`](@ref) which modifies `cube`.
"""
function inject!(cube::AbstractArray{T,3}, kernel, pos; kwargs...) where T
    # All zeros position angles is actually 90° parallactic angle
    angles = Zeros(size(cube, 1))
    return inject!(cube, angles, kernel, pos; kwargs...)
end

function inject!(cube::AbstractArray{T,3}, angles::AbstractVector, kernel, pos; A = one(T), kwargs...) where T
    gen = CubeGenerator(cube, angles, kernel; kwargs...)
    return inject!(cube, gen, pos; kwargs...)
end

function inject!(cube::AbstractArray{T,3}, gen::CubeGenerator, pos; A=one(T)) where T
    return gen(cube, pos; A=A)
end

###################

function normalize_psf!(psf::AbstractMatrix, fwhm, factor=1)
    ap = CircularAperture(center(psf), factor * fwhm / 2)
    area = photometry(ap, psf).aperture_sum
    return psf ./= area
end
normalize_psf(psf, fwhm, factor=1) = normalize_psf!(deepcopy(psf), fwhm, factor)
