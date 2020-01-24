using Images
using Statistics

export DataCube,
       angles,
       derotate!,
       derotate

struct DataCube
    cube::Array{<:Number,3}
    angles::Vector{<:Number}
end

function DataCube(m::AbstractMatrix, angles::AbstractVector)
    xy, n = size(m)
    # Check the input size is square
    @assert sqrt(xy) % 1 == 0.0
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
        frame = imrotate(frame, deg2rad(d.angles[i]), axes(frame))
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
