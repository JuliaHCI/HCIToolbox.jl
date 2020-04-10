using ImageTransformations: center
using Photometry
using Distributions
using Statistics

"""
    snrmap(data, fwhm)

Parallel implementation of signal-to-noise ratio (SNR, S/N) applied to each pixel in the input image.

Uses [`snr`](@ref) (small samples penalty) in resolution elements of size `fwhm` across the whole image.

!!! tip
    This code is automatically multi-threaded, so be sure to set `JULIA_NUM_THREADS` before loading your runtime to take advantage of it!
"""
function snrmap(data::AbstractMatrix{T}, fwhm) where T
    out = fill!(similar(data), zero(T))
    width = minimum(size(data)) / 2 - 1.5 * fwhm

    mask = get_annulus_segments(data, fwhm/2 + 2, width, mode=:mask)
    coords = findall(!iszero, mask)

    Threads.@threads for coord in coords
        @inbounds out[coord] = snr(data, coord, fwhm)
    end
    
    return out
end

"""
    snr(data, position, fwhm)

Calculate the signal to noise ratio (SNR, S/N) for a test point at `position` in a residual frame.

Uses the method of Mawet et al. 2014 which includes penalties for small sample statistics. These are encoded by using a student's t-test for calculating the SNR.

!!! note
    SNR is not equivalent to significance.
"""
function snr(data::AbstractMatrix, position, fwhm)
    x, y = position
    cy, cx = center(data)
    separation = sqrt((x - cx)^2 + (y - cy)^2)
    @assert separation > fwhm/2 + 1 "`position` is too close to the frame center"

    θ = 2asin(fwhm/2/separation)
    N = floor(Int, 2π/θ)

    sint, cost = sincos(θ)
    xs = similar(data, N)
    ys = similar(data, N)

    # initial points
    rx = x - cx
    ry = y - cy

    @inbounds for idx in eachindex(xs)
        xs[idx] = rx + cx
        ys[idx] = ry + cy
        rx, ry = cost * rx + sint * ry, cost * ry - sint * rx
    end

    r = fwhm / 2

    apertures = CircularAperture.(xs, ys, r)
    fluxes = aperture_photometry(apertures, data, method=:exact).aperture_sum
    other_elements = @view fluxes[2:end]
    bkg_σ = std(other_elements) # ddof = 1 by default
    return (fluxes[1] - mean(other_elements)) / (bkg_σ * sqrt(1 + 1/(N - 1)))
end

snr(data::AbstractMatrix, idx::CartesianIndex, fwhm) = snr(data, (idx.I[2], idx.I[1]), fwhm)

snr_to_sig(snr, separation, fwhm) = @. quantile(Normal(), cdf(TDist(2π * separation / fwhm.- 2), snr))
sig_to_snr(sig, separation, fwhm) = @. quantile(TDist(2π * separation / fwhm.- 2), cdf(Normal(), sig))
