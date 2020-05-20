using Distributions
using Optim
using SpecialFunctions
using Statistics
using Photometry.Aperture
using Rotations
using CoordinateTransformations
using ImageTransformations
using Interpolations
using StaticArrays

export Kernels, PSFModel, normalize_psf, normalize_psf!

###################

abstract type PSFKernel end

struct PSFModel{K<:Union{PSFKernel, AbstractMatrix}}
    kernel::K
end

(model::PSFModel)(size::Vararg{<:Integer, 2}; A=1, pa=0, location...) = 
    model(Base.OneTo.(size); A=A, pa=pa, location...)

function (model::PSFModel{<:PSFKernel})(idxs; A=1, pa=0, location...)
    ys, xs = idxs
    ctr = _frame_center(idxs)
    x0, y0 = _get_center(ctr, values(location), pa)
    return @. A * model.kernel(x0, y0, xs', ys)
end

function (model::PSFModel{<:AbstractMatrix{T}})(idxs; A=1, pa=0, location...) where T
    ctr = _frame_center(idxs)
    # have to do reversing and flip sign because `warp` moves canvas not image
    pos = _get_center(ctr, values(location), pa) |> reverse
    tform = Translation(_frame_center(model.kernel) - pos)
    return warp(model.kernel, tform, idxs, Linear(), zero(T))
end

###################

function _frame_center(axes)
    map(axes) do axis
        l, u = extrema(axis)
        (u - l) / 2 + 0.5
    end |> reverse |> SVector
end

_frame_center(img::AbstractMatrix) = ImageTransformations.center(img) |> reverse

function _get_center(ctr, pars::NamedTuple{(:x, :y)}, pa=0)
    pos = SVector(pars.x, pars.y)
    return recenter(RotMatrix{2}(deg2rad(-pa)), ctr)(pos)
end

_get_center(ctr, pars::NamedTuple{(:y, :x)}, pa=0) = _get_center((x=pars.x, y=pars.y), pa)

function _get_center(ctr, pars::NamedTuple{(:r, :theta)}, pa=0)
    pos = Polar(float(pars.r), deg2rad(pars.theta - pa))
    return CartesianFromPolar()(pos) + ctr
end

_get_center(ctr, pars::NamedTuple{(:theta, :r)}, pa=0) = _get_center((r=pars.r, theta=pars.theta), pa)
_get_center(ctr, pars::NamedTuple{(:r, :θ)}, pa=0) = _get_center((r=pars.r, theta=pars.θ), pa)
_get_center(ctr, pars::NamedTuple{(:θ, :r)}, pa=0) = _get_center((r=pars.r, theta=pars.θ), pa)

###################


_sqeuclidean(x0, y0, x, y) = (x - x0)^2 + (y - y0)^2

struct Gaussian{T} <: PSFKernel
    fwhm::T
end

(g::Gaussian)(x0, y0, x, y) = exp(-4log(2) * _sqeuclidean(x0, y0, x, y) / g.fwhm^2)

###################

struct Moffat{T} <: PSFKernel
    fwhm::T
end

function (m::Moffat)(x0, y0, x, y)
    hwhm = m.fwhm / 2
    α = 1
    return (1 + _sqeuclidean(x0, y0, x, y) / hwhm^2)^(-α)
end

###################

struct AiryDisk{T} <: PSFKernel
    fwhm::T
end

const rz = 3.8317059702075125 / π

function (a::AiryDisk)(x0, y0, x, y)
    radius = a.fwhm * 1.18677
    r = sqrt(_sqeuclidean(x0, y0, x, y)) / (radius / rz)
    return iszero(r) ? one(T) : (2besselj1(π * r) / (π * r))^2
end

const Kernels = (Gaussian=Gaussian, Normal=Gaussian, Moffat=Moffat, Airy=AiryDisk)

###################

function normalize_psf!(psf::AbstractMatrix, fwhm, factor=1)
    ap = CircularAperture(center(psf), factor * fwhm / 2)
    area = aperture_photometry(ap, psf).aperture_sum
    return psf ./= area
end
normalize_psf(psf, fwhm, factor=1) = normalize_psf!(deepcopy(psf), fwhm, factor)
