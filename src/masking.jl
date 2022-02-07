using ImageTransformations: center

"""
    mask_circle!(::AbstractMatrix, npix; fill=0, center=center(arr))
    mask_circle!(::AbstractArray, npix; fill=0, center=center(arr))

In-place version of [`mask_circle`](@ref)
"""
function mask_circle!(arr::AbstractMatrix{T}, npix; fill=zero(T), center=center(arr)) where T
    @inbounds for idx in CartesianIndices(arr)
        d = sqrt((idx.I[1] - center[1])^2 + (idx.I[2] - center[2])^2)
        if d < npix
            arr[idx] = fill
        end
    end
    return arr
end

function mask_circle!(cube::AbstractArray{T,3}, npix; kwargs...) where T
    @inbounds for i in axes(cube, 3)
        slice = @view cube[:, :, i]
        mask_circle!(slice, npix; kwargs...)
    end
    return cube
end

"""
    mask_circle(::AbstractMatrix, npix; fill=0, center=center(arr))
    mask_circle(::AbstractArray, npix; fill=0, center=center(arr))

Mask the inner-circle of an image with radius `npix` with value `fill`. Note that the input type must be compatible with the fill value's type. If the input is a cube it will mask each frame individually.

# See Also
[`mask_circle!`](@ref)
"""
mask_circle(arr::AbstractMatrix, npix; kwargs...) = mask_circle!(copy(arr), npix; kwargs...)
mask_circle(cube::AbstractArray{T,3}, npix; kwargs...) where T = mask_circle!(copy(cube), npix; kwargs...)

"""
    mask_annulus!(::AbstractMatrix, npix_in, npix_out; fill=0, center=center(arr))

In-place version of [`mask_annulus`](@ref)
"""
function mask_annulus!(arr::AbstractMatrix{T}, npix_in, npix_out; fill = zero(T), center=center(arr)) where T
    @inbounds for idx in CartesianIndices(arr)
        d = sqrt((idx.I[1] - center[1])^2 + (idx.I[2] - center[2])^2)
        if npix_in ≤ d < npix_out
            arr[idx] = fill
        end
    end
    return arr
end


"""
    mask_annulus(::AbstractMatrix, npix_in, npix_out; fill=0, center=center(arr))

Mask an annular region of an image with inner-radius `npix_in`, outer-radius `npix_out` with value `fill`.
Note that the input type must be compatible with the fill value's type.

# See Also
[`mask_annulus!`](@ref)
"""
mask_annulus(arr::AbstractMatrix, npix_in, npix_out; kwargs...) = mask_annulus!(copy(arr), npix_in, npix_out; kwargs...)

"""
    get_annulus_segments(data, inner_radius, width, [nsegments];
                         theta_init=0, scale_factor=1, mode=:index,
                         center=center(data))

Returns indices or values in segments of a centered annulus.

### Modes
* `:mask` - returns a positive boolean mask for indexing
* `:value` - returns the data values indexed with the boolean mask
* `:apply` - returns the input data weighted by the boolean mask
"""
function get_annulus_segments(data, inner_radius, width, nsegments; theta_init=0, scale_factor=1, mode=:mask, center=center(data))
    cx, cy = center
    xs, ys = axes(data)
    dist = @. sqrt((xs - cx)^2 + (ys' - cy)^2)
    ϕ_stride = deg2rad(ceil(Int, 360 / nsegments))
    ϕ = @. atan(ys - cy, xs - cx)
    ϕ_rot = mod2pi.(ϕ)
    outer_radius = @. inner_radius + scale_factor * width

    masks = similar(data, Matrix{Bool}, nsegments)
    for i in 1:nsegments
        ϕ_start = deg2rad(theta_init) + (i - 1) * ϕ_stride
        ϕ_end = ϕ_start + ϕ_stride

        if ϕ_start < 2π && ϕ_end > 2π
            mask = @. (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ ϕ_start) & (ϕ_rot ≤ 2π) | (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ 0) & (ϕ_rot < ϕ_end - 2π)
        elseif ϕ_start ≥ 2π && ϕ_end > 2π
            mask = @. (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ ϕ_start - 2π) & (ϕ_rot < ϕ_end - 2π)
        else
            mask = @. (dist ≥ inner_radius) & (dist < outer_radius) & (ϕ_rot ≥ ϕ_start) & (ϕ_rot < ϕ_end)
        end
        @inbounds masks[i] = mask
    end

    return _convert_mask.(Val(mode), masks, (data,))
end


get_annulus_segments(data, inner_radius, width; kwargs...) = get_annulus_segments(data, inner_radius, width, 1; kwargs...) |> only


_convert_mask(::Val{:mask}, mask, data) = mask
_convert_mask(::Val{:value}, mask, data) = data[mask]
_convert_mask(::Val{:apply}, mask, data) = data .* mask

