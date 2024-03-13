
using LossFunctions
using Optim
using PaddedViews

"""
    scale(cube::AbstractArray{T,3}, scales)

Linearly stretch each frame in `cube` by the corresponding scale in `scales`. Uses bilinear interpolaiton internally. The output size can be specified or else we will choose the smallest size that contains the largest stretch.
"""
function scale(cube::AbstractArray{T,3}, scales) where T
    out = similar(cube)
    Threads.@threads for i in axes(cube, 3)
        frame = @view cube[:, :, i]
        out[:, :, i] .= scale(frame, scales[i])
    end
    return out
end

"""
    scale(frame::AbstractMatrix, scale)

Linearly stretch `frame` with the ratio `scale`. Uses bilinear interpolation internally.
"""
scale(frame::AbstractMatrix, scale) = cropview(imresize(frame; ratio=scale), size(frame); force=true, verbose=false)


"""
    invscale(cube::AbstractArray{T,3}, scales)

Linearly contract each frame in `cube` by the corresponding scale in `scales`. Uses bilinear interpolaiton internally.
"""
function invscale(cube::AbstractArray{T,3}, scales) where T
    out = similar(cube)
    Threads.@threads for i in axes(cube, 3)
        out[:, :, i] .=  invscale(cube[:, :, i], scales[i])
    end
    return out
end


"""
    invscale(frame::AbstractMatrix, scale)

Linearly contract `frame` with the ratio `scale`. Uses bilinear interpolation internally.
"""
function invscale(frame::AbstractMatrix{T}, scale) where T
    out = imresize(frame; ratio=inv(scale))
    size(out) == size(frame) && return out
    offset = div.(size(frame) .- size(out), 2) .+ 1
    return PaddedView(zero(T), out, size(frame), offset)
end

"""
    scale_and_stack(spcube::AbstractArray{T,4}, scales)

Given a 4-D spectral ADI (SDI) tensor this function scales each temporal slice according to `scales` and then concatenates into a cube with `nλ * nf` frames.
"""
function scale_and_stack(spcube::AbstractArray{T,4}, scales) where T
    nx, ny, nλ, nf = size(spcube)
    out = similar(spcube, nx, ny, nλ * nf)
    Threads.@threads for n in axes(spcube, 4)
        slice = (n - 1) * nλ + firstindex(spcube, 4):n * nλ
        frame = @view spcube[:, :, :, n]
        out[:, :, slice] .= scale(frame, scales)
    end

    return out
end

"""
    invscale_and_collapse(stack_cube::AbstractArray{T,3}, scales; kwargs...)

Given an SDI tensor that has been stacked into a cube, invscales each spectral slice and combines with [`collapse`](@ref). The output will be cropped to `size`.
"""
function invscale_and_collapse(stack_cube::AbstractArray{T,3}, scales; kwargs...) where T
    nλ = length(scales)
    n = Base.size(stack_cube, 3) ÷ nλ
    out = similar(stack_cube, size(stack_cube)[1:2]..., n)
    Threads.@threads for n in axes(out, 3)
        slice = (n - 1) * nλ + firstindex(out, 3):n * nλ
        cube = @view stack_cube[:, :, slice]
        out[:, :, n] .= collapse(invscale(cube, scales); kwargs...)
    end
    return out
end

"""
    scale_list(wavelengths)

Returns a list of scaling factors for aligning SDI tensors from a list of wavelengths.

# Examples

```jldoctest
julia> scale_list([0.5, 2, 4])
3-element Vector{Float64}:
 8.0
 2.0
 1.0
```
"""
scale_list(wavelengths) = maximum(wavelengths) ./ wavelengths

function optimize_scale_list(spcube::AbstractArray{T,4}, scales, amps=ones(size(spcube, 3)); kwargs...) where T
    # get temporal median first
    spframe = median(spcube, dims=4)[:, :, :, 1]
    return optimize_scale_list(spframe, scales; kwargs...)
end

function optimize_scale_list(spframe::AbstractArray{T,3}, scales, amps=ones(size(spframe, 3)); mask=trues(size(spframe, 1), size(spframe, 2))) where T
    reference_frame = @view spframe[:, :, end]
    N_wl = size(spframe, 3)
    best_scales = ones(N_wl)
    best_flux = ones(N_wl)
    for wl_idx in axes(spframe, 3)[begin:end - 1]
        current_frame = @view spframe[:, :, wl_idx]
        func(X) = _scale_opt_func(current_frame, reference_frame, X[begin], X[end], mask)
        P0 = T[scales[wl_idx], amps[wl_idx]]
        result = optimize(func, P0, NelderMead(); autodiff=:forward)
        @info "Finished optimizing (wl=$wl_idx/$N_wl)" result
        X = Optim.minimizer(result)
        best_scales[wl_idx] = X[begin]
        best_flux[wl_idx] = X[end]
    end
    return (;scales=best_scales, fluxes=best_flux)
end

function _scale_opt_func(frame, reference, scale, amp=1, mask=trues(frame))
    scale < 1 && return Inf
    scaled_frame = amp .* HCIToolbox.scale(frame, scale)
    # get loss
    return sum(L2DistLoss(),  mask .* (scaled_frame .- reference))
end