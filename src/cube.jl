using Images: imrotate, center
using Statistics

export DataCube,
       angles,
       derotate!,
       derotate,
       mask!,
       mask

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

angles(d::DataCube) = d.angles

Base.:(==)(d1::DataCube, d2::DataCube) =  d1.cube == d2.cube && d1.angles == d2.angles

function derotate!(d::DataCube)
    @inbounds for i in axes(d.cube, 3)
        frame = @view d.cube[:, :, i]
        d.cube[:, :, i] .= imrotate(frame, -deg2rad(d.angles[i]), axes(frame))
    end
    return d
end

derotate(d::DataCube) = derotate!(deepcopy(d))

function Statistics.mean(d::DataCube)
    out = mean(d.cube, dims = 3)
    return out[:, :, 1]
end

function Statistics.median(d::DataCube)
    out = median(d.cube, dims = 3)
    return out[:, :, 1]
end

function mask!(arr::AbstractMatrix, npix)
    yy = axes(arr, 1)
    xx = axes(arr, 2)
    yc, xc = center(arr)
    d = @. sqrt((xx' - xc)^2 + (yy - yc)^2)
    arr[d .< npix] .= NaN
    return arr
end

mask(arr::AbstractMatrix, npix) = mask!(deepcopy(arr), npix)
