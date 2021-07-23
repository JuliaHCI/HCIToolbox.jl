
using PaddedViews

"""
    scale(cube::AbstractArray{T,3}, scales[, size])

Linearly stretch each frame in `cube` by the corresponding scale in `scales`. Uses bilinear interpolaiton internally. The output size can be specified or else we will choose the smallest size that contains the largest stretch.
"""
function scale(cube::AbstractArray{T, 3}, scales, size) where T
    out = similar(cube, Base.size(cube, 1), size...)
    Threads.@threads for i in axes(cube, 1)
        frame = @view cube[i, :, :]
        out[i, :, :] .= scale(frame, scales[i], size)
    end
    return out
end

scale(cube::AbstractArray{T, 3}, scales) where {T} = scale(cube, scales, stretch_size(cube, scales))

stretch_size(cube, scales) = stretch_size((size(cube, 2), size(cube, 3)), scales)

function stretch_size(size::Tuple, scales)
    max_size = ceil.(Int, maximum(scales) .* size)
    return check_size(size, max_size)
end

scale(frame::AbstractMatrix, scale) = imresize(frame; ratio=scale)

"""
    scale(frame::AbstractMatrix, scale[, size])

Linearly stretch `frame` with the ratio `scale`, padding symmetrically with zeros to match `out_size`. Uses bilinear interpolation internally. If `size` is specified, will return a padded view.
"""
function scale(frame::AbstractMatrix{T}, scale, out_size) where T
    res_frame = imresize(frame; ratio=scale)
    init_pos = (out_size .- size(res_frame)) .÷ 2 .+ 1
    return PaddedView(zero(T), res_frame, out_size, init_pos)
end

"""
    invscale(cube::AbstractArray{T,3}, scales[, size])

Linearly contract each frame in `cube` by the corresponding scale in `scales`. Uses bilinear interpolaiton internally. The output size can be specified or else we will choose the smallest size that contains the largest stretch.
"""
function invscale(cube::AbstractArray{T, 3}, scales, size) where T
    out = similar(cube, Base.size(cube, 1), size...)
    Threads.@threads for i in axes(cube, 1)
        out[i, :, :] .=  invscale(cube[i, :, :], scales[i], size)
    end
    return out
end

invscale(cube::AbstractArray{T, 3}, scales) where {T} = invscale(cube, scales, shrink_size(cube, scales))

shrink_size(cube, scales) = shrink_size((size(cube, 2), size(cube, 3)), scales)

function shrink_size(size::Tuple, scales)
    min_size = ceil.(Int, size ./ maximum(scales))
    return check_size(size, min_size)
end


"""
    invscale(frame::AbstractMatrix, scale[, size])

Linearly contract `frame` with the ratio `scale`. Uses bilinear interpolation internally. If `size` is specified, will return a crop view.
"""
invscale(frame::AbstractMatrix, scale) = imresize(frame; ratio=inv(scale))
invscale(frame::AbstractMatrix, scale, size) = cropview(imresize(frame; ratio=inv(scale)), size; force=true, verbose=false)

"""
    scale_and_stack(spcube::AbstractArray{T,4}, scales)

Given a 4-D spectral ADI (SDI) tensor, scales each, temporal slice according to `scales` and then concatenates into a cube with `nλ * nf` frames.
"""
function scale_and_stack(spcube::AbstractArray{T, 4}, scales) where T
    nλ, nf, ny, nx = size(spcube)
    frame_size = (ny, nx)
    max_scale = maximum(scales)
    out_size = check_size(frame_size, ceil.(Int, max_scale .* frame_size))
    out = similar(spcube, nλ * nf, out_size...)
    Threads.@threads for n in axes(spcube, 2)
        slice = (n - 1) * nλ + firstindex(spcube, 2):n * nλ
        frame = @view spcube[:, n, :, :]
        out[slice, :, :] .= scale(frame, scales)
    end

    return out
end

"""
    invscale_and_collapse(stack_cube::AbstractArray{T,3}, scales, size; kwargs...)

Given an SDI tensor that has been stacked into a cube, invscales each spectral slice and combines with [`collapse`](@ref). The output will be cropped to `size`.
"""
function invscale_and_collapse(stack_cube::AbstractArray{T, 3}, scales, size; kwargs...) where T
    nλ = length(scales)
    n = Base.size(stack_cube, 1) ÷ nλ
    out = similar(stack_cube, n, size...)
    Threads.@threads for n in axes(out, 1)
        slice = (n - 1) * nλ + firstindex(out, 1):n * nλ
        cube = @view stack_cube[slice, :, :]
        out[n, :, :] .= collapse(invscale(cube, scales, size); kwargs...)
    end
    return out
end

"""
    scale_list(wavelengths)

Returns a list of scaling factors for aligning SDI tensors from a list of wavelengths. Note that you could equivalently pass in a spectral template that has already been evaluated on the same wavelength grid as the data.
"""
scale_list(wavelengths) = maximum(wavelengths) ./ wavelengths

