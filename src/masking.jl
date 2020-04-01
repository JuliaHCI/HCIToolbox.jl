using ImageTransformations: center

"""
    mask_circle!(::AbstractMatrix, npix; fill=NaN)

In-place version of [`mask_circle`](@ref)
"""
function mask_circle!(arr::AbstractMatrix, npix; fill = NaN)
    y, x = axes(arr)
    yc, xc = center(arr)
    d = @. sqrt((x' - xc)^2 + (y - yc)^2)
    @. arr[d < npix] = fill
    return arr
end

"""
    mask_circle(::AbstractMatrix, npix; fill=NaN)

Mask the inner-circle of an image with radius `npix` with value `fill`. Note that the input type must be compatible with the fill value's type.

# See Also
* [`mask_circle!`](@ref)
"""
mask_circle(arr::AbstractMatrix{<:AbstractFloat}, npix; fill = NaN) = mask_circle!(deepcopy(arr), npix, fill = fill)

"""
    mask_annulus!(::AbstractMatrix, npix_in, npix_out; fill=NaN)

In-place version of [`mask_annulus`](@ref)
"""
function mask_annulus!(arr::AbstractMatrix, npix_in, npix_out; fill = NaN)
    y, x = axes(arr)
    yc, xc = center(arr)
    d = @. sqrt((x' - xc)^2 + (y - yc)^2)
    @. arr[npix_in ≤ d < npix_out] = fill
    return arr
end


"""
    mask_annulus(::AbstractMatrix, npix_in, npix_out; fill=NaN)

Mask an annular region of an image with inner-radius `npix_in`, outer-radius `npix_out` with value `fill`.
Note that the input type must be compatible with the fill value's type.

# See Also
* [`mask_annulus!`](@ref)
"""
mask_annulus(arr::AbstractMatrix{<:AbstractFloat}, npix_in, npix_out; fill = NaN) = mask_annulus!(deepcopy(arr), npix_in, npix_out, fill = fill)
