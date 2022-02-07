using Statistics
using FillArrays
using Rotations
using CoordinateTransformations
using ImageTransformations
using Interpolations
using LinearAlgebra
using StaticArrays

"""
    inject(frame::AbstractMatrix, psf, [angle]; x, y, amp=1, center=center(frame), kwargs...)

Injects the `psf` into `frame` at the given position.

# Examples

`inject` works with two types of PSF models: matrices and synthetic models. 

## Matrices

If you pass an `AbstractMatrix` to `inject`, you can optionally specify `degree` and `fill` as keyword arguments. By default, `degree=Interpolations.Linear()` and `fill=0`. These are used to create an `Interpolations.AbstractExtrapolation` type which can be arbitrarily transformed. From here, the `x`, `y`, and `amp` arguments will determine the position and will be optionally rotated `angle` degrees counter-clockwise around the `center`.

```jldoctest
julia> inject(zeros(5, 5), ones(1, 1); x=4, y=3, amp=2)
5×5 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  2.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> inject(zeros(5, 5), ones(3, 3), 90; x=4, y=3, amp=2)
5×5 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  2.0  2.0  2.0
 0.0  0.0  2.0  2.0  2.0
 0.0  0.0  2.0  2.0  2.0
 0.0  0.0  0.0  0.0  0.0
```

## Synthetic Models

The synthetic models from [`PSFModels.jl`](https://github.com/JuliaAstro/PSFModels.jl) can easily be used, too. In this case, any additional keyword arguments are directly passed to the given model-

```jldoctest
julia> using PSFModels

julia> inject(zeros(5, 5), gaussian; x=4, y=3, amp=2, fwhm=2)
5×5 Matrix{Float64}:
 0.000244141  0.00195312  0.00390625  0.00195312  0.000244141
 0.0078125    0.0625      0.125       0.0625      0.0078125
 0.0625       0.5         1.0         0.5         0.0625
 0.125        1.0         2.0         1.0         0.125
 0.0625       0.5         1.0         0.5         0.0625

julia> inject(zeros(5, 5), airydisk, 45; x=4, y=3, amp=2, fwhm=2, ratio=0.3)
5×5 Matrix{Float64}:
 0.042946   0.0944566   0.0741111  0.06456   0.0914308
 0.0944566  0.0191043   0.054543   0.108174  0.00031632
 0.0741111  0.054543    0.929442   1.29573   0.307196
 0.06456    0.108174    1.29573    1.76798   0.470669
 0.0914308  0.00031632  0.307196   0.470669  0.0620712
```

"""
inject(frame::AbstractMatrix, args...; kwargs...) = inject!(copy(frame), args...; kwargs...)

function inject!(frame::AbstractMatrix, kernel::AbstractMatrix{T}, args...; degree=Linear(), fill=zero(T), kwargs...) where T
    etp = ImageTransformations.box_extrapolation(kernel; method=degree, fillvalue=fill)
    return inject!(frame, etp, args...; kwargs...)
end

function inject!(frame::AbstractMatrix{T}, kernel::AbstractExtrapolation, angle=0; x, y, amp=one(T), center=center(frame), inds=CartesianIndices(frame)) where T
    if iszero(angle)
        idxmap = identity
    else
        idxmap = recenter(RotMatrix{2}(-deg2rad(angle)), center)
    end
    tform = Translation(ImageTransformations.center(kernel) .- (x, y)) ∘ idxmap
    @inbounds for idx in inds
        point = tform(SVector(idx.I))
        frame[idx] += amp * kernel(point...)
    end
    return frame
end

function inject!(frame::AbstractMatrix{T}, psfmodel, angle=0; center=center(frame), inds=CartesianIndices(frame), kwargs...) where T
    if iszero(angle)
        idxmap = identity
    else
        idxmap = recenter(RotMatrix{2}(-deg2rad(angle)), center)
    end
    @inbounds for idx in inds
        point = idxmap(SVector(idx.I))
        frame[idx] += psfmodel(T, point; kwargs...)
    end
    return frame
end

################################

"""
    inject(cube, psf, [angles]; x, y, amp=1, kwargs...)


Injects the matrix `psf` into each frame of `cube` at the given position.

If `angles` are provided, each position will be rotated by the given parallactic angles in degrees, centered at `center`, which is the center of each frame, by default.
"""
inject(cube::AbstractArray{T,3}, args...; kwargs...) where T =
    inject!(copy(cube), args...; kwargs...)

"""
    inject!(cube, psf, [angles]; kwargs...)

In-place version of [`inject`](@ref) which modifies `cube`.
"""
function inject!(cube::AbstractArray{T,3}, kernel, angles=Zeros(size(cube, 3)); kwargs...) where T
    # All zeros position angles is actually 90° parallactic angle
    @inbounds for idx in axes(cube, 3)
        frame = @view cube[:, :, idx]
        inject!(frame, kernel, angles[idx]; kwargs...)
    end
    return cube
end

function inject!(cube::AnnulusView{T}, kernel, angles=Zeros(size(cube, 3)); kwargs...) where T
    # All zeros position angles is actually 90° parallactic angle
    @inbounds for idx in cube.indices[2]
        # relies on the fact that the underlying data can still be modified
        frame = @view cube[:, :, idx]
        inject!(frame, kernel, angles[idx]; inds=cube.indices[1], kwargs...)
    end
    return cube
end

function inject!(cube::MultiAnnulusView{T}, kernel, angles=Zeros(size(cube, 3)); kwargs...) where T
    # All zeros position angles is actually 90° parallactic angle
    @inbounds for annidx in cube.indices
        for idx in annidx[2]
            inds = annidx[1]
            # relies on the fact that the underlying data can still be modified
            frame = @view cube[:, :, idx]
            inject!(frame, kernel, angles[idx]; inds, kwargs...)
        end
    end
    return cube
end
