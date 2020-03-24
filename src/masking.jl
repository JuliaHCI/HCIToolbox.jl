using ImageTransformations: center

"""
mask_inner!(::AbstractMatrix, npix; fill=NaN)

In-place version of [`mask_inner`](@ref)
"""
function mask_inner!(arr::AbstractMatrix{T}, npix; fill = NaN) where {T <: AbstractFloat}
    yy = axes(arr, 1)
    xx = axes(arr, 2)
    yc, xc = center(arr)
    d = @. sqrt((xx' - xc)^2 + (yy - yc)^2)
    @. arr[d < npix] = T(fill)
    return arr
end

"""
mask_inner(::AbstractMatrix, npix; fill=NaN)

Mask the inner-circle of an image with radius `npix` with value `fill`. Note that the input type must be compatible with the fill value's type.

# See Also
* [`mask_inner!`](@ref)
"""
mask_inner(arr::AbstractMatrix{<:AbstractFloat}, npix; fill = NaN) = mask_inner!(deepcopy(arr), npix, fill = fill)
