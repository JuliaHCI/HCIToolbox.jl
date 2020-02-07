using Images: imrotate, center
using Statistics

export DataCube,
       angles,
       derotate!,
       derotate,
       mask!,
       mask,
       coadd

struct DataCube
    cube::Array{<:Number,3}
    angles::Vector{<:Number}
end

DataCube(c::AbstractArray{T,3}) where {T <: Number} = DataCube(c, zeros(T, size(c, 3)))

function DataCube(m::AbstractMatrix, angles::AbstractVector)
    xy, n = size(m)
    # Check the input size is square
    @assert isinteger(sqrt(xy))
    newsize = (Int(sqrt(xy)), Int(sqrt(xy)), n)
    cube = reshape(m, newsize)
    return DataCube(cube, angles)
end

function Base.Matrix(d::DataCube)
    x, y, n = size(d.cube)
    return reshape(d.cube, (x * y, n))
end

cube(dc::DataCube) = dc.cube
angles(d::DataCube) = d.angles

Base.length(dc::DataCube) = length(d.cube)
Base.size(dc::DataCube) = size(dc.cube)
Base.size(dc::DataCube, dims) = size(dc.cube, dims)

nframes(dc::DataCube) = size(dc, 3)

Base.:(==)(d1::DataCube, d2::DataCube) =  d1.cube == d2.cube && d1.angles == d2.angles

"""
    derotate!(::DataCube)

In-place version of [`derotate`](@ref)
"""
function derotate!(d::DataCube)
    @inbounds for i in axes(d.cube, 3)
        frame = @view d.cube[:, :, i]
        d.cube[:, :, i] .= imrotate(frame, -deg2rad(d.angles[i]), axes(frame))
    end
    return d
end

"""
    derotate(::DataCube)

De-rotates a [`DataCube`](@ref) using its internal angles.
"""
derotate(d::DataCube) = derotate!(deepcopy(d))

"""
    Statistics.mean(::DataCube)

Collapses a [`DataCube`](@ref) by finding the mean frame. Note: does not derotate.
"""
function Statistics.mean(d::DataCube)
    out = mean(d.cube, dims = 3)
    return out[:, :, 1]
end

"""
    Statistics.median(::DataCube)

Collapses a [`DataCube`](@ref) by finding the median frame. Note: does not derotate.
"""
function Statistics.median(d::DataCube)
    out = median(d.cube, dims = 3)
    return out[:, :, 1]
end


"""
    mask!(::AbstractMatrix, npix)

In-place version of [`mask`](@ref)
"""
function mask!(arr::AbstractMatrix, npix)
    yy = axes(arr, 1)
    xx = axes(arr, 2)
    yc, xc = center(arr)
    d = @. sqrt((xx' - xc)^2 + (yy - yc)^2)
    arr[d .< npix] .= NaN
    return arr
end

"""
    mask(::AbstractMatrix, npix)

Mask the inner-circle of an image with radius `npix`.
"""
mask(arr::AbstractMatrix, npix) = mask!(deepcopy(arr), npix)

