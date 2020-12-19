using ImageTransformations: center
using Compat

"""
    mask_circle!(::AbstractMatrix, npix; fill=0)
    mask_circle!(::AbstractArray, npix; fill=0)

In-place version of [`mask_circle`](@ref)
"""
function mask_circle!(arr::AbstractMatrix, npix; fill = 0)
    y, x = axes(arr)
    yc, xc = center(arr)
    d = @. sqrt((x' - xc)^2 + (y - yc)^2)
    @. arr[d < npix] = fill
    return arr
end

function mask_circle!(cube::AbstractArray{T, 3}, npix; fill=0) where T
    @inbounds for i in axes(cube, 1)
        slice = @view cube[i, :, :]
        mask_circle!(slice, npix; fill=fill)
    end
    return cube
end

"""
    mask_circle(::AbstractMatrix, npix; fill=0)
    mask_circle(::AbstractArray, npix; fill=0)

Mask the inner-circle of an image with radius `npix` with value `fill`. Note that the input type must be compatible with the fill value's type. If the input is a cube it will mask each frame individually.

# See Also
[`mask_circle!`](@ref)
"""
mask_circle(arr::AbstractMatrix, npix; fill = 0) = mask_circle!(deepcopy(arr), npix, fill = fill)
mask_circle(cube::AbstractArray{T, 3}, npix; fill = 0) where T = mask_circle!(deepcopy(cube), npix, fill=fill)

"""
    mask_annulus!(::AbstractMatrix, npix_in, npix_out; fill=NaN)

In-place version of [`mask_annulus`](@ref)
"""
function mask_annulus!(arr::AbstractMatrix, npix_in, npix_out; fill = 0)
    y, x = axes(arr)
    yc, xc = center(arr)
    d = @. sqrt((x' - xc)^2 + (y - yc)^2)
    @. arr[npix_in ≤ d < npix_out] = fill
    return arr
end


"""
    mask_annulus(::AbstractMatrix, npix_in, npix_out; fill=0)

Mask an annular region of an image with inner-radius `npix_in`, outer-radius `npix_out` with value `fill`.
Note that the input type must be compatible with the fill value's type.

# See Also
[`mask_annulus!`](@ref)
"""
mask_annulus(arr::AbstractMatrix, npix_in, npix_out; fill = 0) = mask_annulus!(deepcopy(arr), npix_in, npix_out, fill = fill)

"""
    get_annulus_segments(data, inner_radius, width, [nsegments];
                         theta_init=0, scale_factor=1, mode=:index)

Returns indices or values in segments of a centered annulus.

### Modes
* `:mask` - returns a positive boolean mask for indexing
* `:value` - returns the data values indexed with the boolean mask
* `:apply` - returns the input data weighted by the boolean mask
"""
function get_annulus_segments(data, inner_radius, width, nsegments; theta_init=0, scale_factor=1, mode=:mask)
    cy, cx = center(data)
    ys, xs = axes(data)
    dist = @. sqrt((xs' - cx)^2 + (ys - cy)^2)
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


@compat get_annulus_segments(data, inner_radius, width; theta_init=0, scale_factor=1, mode=:mask) = get_annulus_segments(data, inner_radius, width, 1; theta_init=theta_init, scale_factor=scale_factor, mode=mode) |> only


_convert_mask(::Val{:mask}, mask, data) = mask
_convert_mask(::Val{:value}, mask, data) = data[mask]
_convert_mask(::Val{:apply}, mask, data) = data .* mask

