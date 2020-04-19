using Distributions
using ImageTransformations: center
using Optim
using SpecialFunctions
using Statistics
using Photometry.Aperture

export Models, fit, synth_psf, fit_psf, normalize_psf, normalize_psf!


abstract type PSFModel{T} end

###################

struct Gaussian{T} <: PSFModel{T}
    x0::T
    y0::T
    fwhm::T
    A::T
end

Gaussian(x0, y0, fwhm, A=1) = Gaussian(promote(x0, y0, fwhm, A)...)
(g::Gaussian)(x, y) = g.A * pdf(MvNormal([g.x0, g.y0], g.fwhm / (2sqrt(2log(2)))), [x, y])

###################

struct Moffat{T} <: PSFModel{T}
    x0::T
    y0::T
    fwhm::T
    A::T
end

Moffat(x0, y0, fwhm, A=1) = Moffat(promote(x0, y0, fwhm, A)...)
function (m::Moffat)(x, y)
    Γ = m.fwhm / 2
    α = 1
    return m.A * (1 + ((x - m.x0)^2 + (y - m.y0)^2)^2 / Γ^2)^(-α)
end

###################

struct AiryDisk{T} <: PSFModel{T}
    x0::T
    y0::T
    fwhm::T
    A::T
end

const rz = 3.8317059702075125 / π

AiryDisk(x0, y0, fwhm, A=1) = AiryDisk(promote(x0, y0, fwhm, A)...)
function (a::AiryDisk)(x, y)
    radius = a.fwhm * 1.18677
    r = sqrt((x - a.x0)^2 + (y - a.y0)^2) / (radius / rz)
    z = iszero(r) ? r : (2besselj1(π * r) / (π * r))^2
    return a.A * z
end

###################

const Models = (gaussian=Gaussian, moffat=Moffat, airy=AiryDisk)

function synth_psf(m::PSFModel{T}, shape) where T
    cy, cx = @. shape ./ 2 + 0.5
    psf = Matrix{T}(undef, shape)
    for idx in CartesianIndices(psf)
        y, x = idx.I
        psf[idx] = m(x - cx, y - cy)
    end
    return psf
end


function fit(M::Type{<:PSFModel}, data::AbstractMatrix{T}, method=LBFGS()) where T
    # χ2 loss function
    loss(P) = mapreduce((y_pred, y) -> (y_pred - y)^2, +, synth_psf(M(P...), size(data)), data)
    P0 = Float64[0, 0, 1, maximum(data)]
    opt = optimize(loss, P0, method; autodiff=:forward)
    @info opt
    return M(Optim.minimizer(opt)...)
end

function fit_psf(M::Type{<:PSFModel}, data::AbstractMatrix{T}, method=LBFGS(); normalize=true) where T
    model = fit(M, data, method)
    psf = synth_psf(model, size(data))

    # optional normalization
    normalize && normalize_psf!(psf, model.fwhm)
    return psf, model
end

function normalize_psf!(psf::AbstractMatrix, fwhm, factor=1)
    ap = CircularAperture(center(psf), factor * fwhm / 2)
    area = aperture_photometry(ap, psf).aperture_sum
    return psf ./= area
end
normalize_psf(psf, fwhm, factor=1) = normalize_psf!(deepcopy(psf), fwhm, factor)
