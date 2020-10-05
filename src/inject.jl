using Statistics
using Photometry.Aperture
using Rotations
using CoordinateTransformations
using ImageTransformations
using Interpolations
using StaticArrays

export Kernels, construct, normalize_psf, normalize_psf!

"""
This module contains some synthetic PSF kernels.

Available kernels
- [`Kernels.Gaussian`](@ref)/[`Kernels.Normal`](@ref)
- [`Kernels.Moffat`](@ref)
- [`Kernels.AiryDisk`](@ref)

# Examples

## Fitting a PSF Model

Here is a quick example fitting a model PSF to data, retrieving a [`Kernels.PSFKernel`](@ref). This example uses [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl) and [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), so please see their documentation for further questions. There are many ways to fit a model, so the important things to recognize are how [`construct`](@ref) and the chosen `PSFModel` integrate into a standard-looking MLE optimization.

```jldoctest; output=false
# Create noisy Airy disk (slightly off-center) with A=24 and FWHM=10
img = construct(Kernels.AiryDisk(10), (101, 101),
                A=24, r=0.2, theta=-10) .+ randn(101, 101)

# Set up optimization problem: linear regression
using LossFunctions, Optim
loss(y_pred) = value(L2DistLoss(), img, y_pred, AggMode.Sum()) # least-squares loss
# if image is pre-centered, don't _need_ to fit position
# use logarithmic transform to make sure fhwm and A are positive
function objective(X)
    log_fwhm, log_A, x, y = X
    img_pred = construct(Kernels.AiryDisk(exp(log_fwhm)), size(img);
                         A=exp(log_A), x=x, y=y)
    return loss(img_pred)
end
# optimize using NelderMead
X0 = Float64[0, 0, 51, 51]
res = optimize(objective, X0, NelderMead())
# OR leverage autodiff and higher order methods like LBFGS and Newton's method
res_ad = optimize(objective, X0, Newton(); autodiff=:forward)

# Set up the best fitting model
fhwm_mle, psf_A_mle = exp.(Optim.minimizer(res_ad)[1:2])
psf_model = Kernels.AiryDisk(fhwm_mle)
```
"""
module Kernels

using SpecialFunctions

"""
    Kernels.PSFKernel <: Function

A kernel used for defining a PSF.

# Interface

Right now, each kernel has one function that returns the amplitude of the kernel given a Euclidean distance. The amplitude should be scaled appropriately such that the maximum value is 1.
```julia
(kernel::PSFKernel)(dist::Number)::Number
```
"""
abstract type PSFKernel <: Function end

"""
    Kernels.Gaussian(fwhm)
    Kernels.Normal(fwhm)

A Gaussian kernel

```math
K(d) = \\exp\\left[-4\\ln{2}\\left(\\frac{d}{\\Gamma}\\right)^2\\right]
```

```
  ┌────────────────────────────────────────┐
1 │                  .'::                  │ Gaussian(x)
  │                 .: :'.                 │
  │                 :  : :                 │
  │                :'  : ':                │
  │                :   :  :                │
  │               .'   :  ':               │
  │               :    :   :               │
  │              .:    :   '.              │
  │              :     :    :              │
  │             .'     :    ':             │
  │             :      :     :             │
  │            .'      :     ':            │
  │           .'       :      ':           │
  │          .'        :       ':          │
0 │.........''         :         ':........│
  └────────────────────────────────────────┘
  -2fwhm                               2fwhm
```
"""
struct Gaussian{T} <: PSFKernel
    fwhm::T
end

(g::Gaussian)(dist) = exp(-4 * log(2) * (dist / g.fwhm)^2)

# Alias Normal -> Gaussian
const Normal = Gaussian
@doc (@doc Gaussian) Normal


"""
    Kernels.Moffat(fwhm)

A Moffat kernel

```math
K(d) = \\left[1 + \\left(\\frac{d}{\\Gamma/2}\\right)^2 \\right]^{-1}
```

```
  ┌────────────────────────────────────────┐
1 │                  .::.                  │ Moffat(x)
  │                  : ::                  │
  │                 :' :':                 │
  │                 :  : :.                │
  │                :   :  :                │
  │               .:   :  :.               │
  │               :    :   :               │
  │              .:    :   '.              │
  │              :     :    '.             │
  │             :      :     :.            │
  │            :       :      :.           │
  │          .'        :       ':          │
  │       ..''         :         ':.       │
  │ ....:''            :            ''.... │
0 │''                  :                  '│
  └────────────────────────────────────────┘
  -2fwhm                               2fwhm
```
"""
struct Moffat{T} <: PSFKernel
    fwhm::T
end

function (m::Moffat)(dist)
    hwhm = m.fwhm / 2
    return inv(1 + (dist / hwhm)^2)
end

"""
    Kernels.AiryDisk(fwhm)

An Airy disk. Guaranteed to work even at `r=0`.

```math
K(r) = \\left[\\frac{2J_1\\left(\\pi r \\right)}{\\pi r}\\right]^2 \\quad r \\approx \\frac{d}{0.97\\Gamma}
```

```
  ┌────────────────────────────────────────┐
1 │                  .'::                  │ AiryDisk(x)
  │                 .' :':                 │
  │                 :  : :                 │
  │                :'  : ':                │
  │                :   :  :                │
  │               .'   :  ':               │
  │               :    :   :               │
  │              .'    :   ':              │
  │              :     :    :              │
  │             .:     :    '.             │
  │             :      :     :             │
  │            .'      :     ':            │
  │            :       :      :.           │
  │           :        :       :           │
0 │..........:'        :        '..........│
  └────────────────────────────────────────┘
  -2fwhm                               2fwhm
```
"""
struct AiryDisk{T} <: PSFKernel
    fwhm::T
end

const rz = 3.8317059702075125 / π

function (a::AiryDisk)(dist)
    radius = a.fwhm * 1.18677
    r = dist / (radius / rz)
    return iszero(r) ? 1.0 : (2besselj1(π * r) / (π * r))^2
end

end # module Kernels

using .Kernels

###################

"""
    construct(kernel, size; A=1, pa=0, location...)
    construct(kernel, indices; A=1, pa=0, location...)

Constructs a synthetic PSF using the given kernel function.

The kernel functions are expected to take the distance from the center of the PSF and return the density. [`Kernels`](@ref) contains common optical PSFs. These kernels are normalized such that the peak amplitude is 1.

If `indices` are provided they will be used as the input grid. If `size` is provided the PSF will have the given size. It will have amplitude `A` and will be located at the given `location` (explained below) potentially rotated by an additional `pa` degrees clockwise. If no `location` is given, will assume the center of the frame.

### Coordinate System
* `x, y` - Parsed as distance from the bottom-left corner of the image. Pixel convention is that `(1, 1)` is the center of the bottom-left pixel increasing right and up.
* `r, theta` or `r, θ` - Parsed as polar coordinates from the center of the image. `theta` is a position angle.
"""
function construct(kernel::Function,
                   idxs::Tuple{<:AbstractVector, <:AbstractVector};
                   A=1,
                   pa=0,
                   location...)
    T = length(location) > 0 ? mapreduce(typeof, promote_type, values(location)) : eltype(A)
    S = float(T)
    ys, xs = idxs
    ctr = _frame_center(idxs)
    x0, y0 = _get_location(ctr, values(location), pa)
    return @. S(A * kernel(sqrt((float(xs') - x0)^2 + (float(ys) - y0)^2)))
end

"""
    construct(::AbstractMatrix, size; A=1, pa=0, degree=Linear(), location...)
    construct(::AbstractMatrix, indices; A=1, pa=0, degree=Linear(), location...)

Constructs a PSF at the given location using the given matrix via bilinear interpolation.

If `indices` are provided they will be used as the input grid. If `size` is provided the PSF will have the given size. It will have amplitude `A` and will be located at the given `location` (explained below) potentially rotated by an additional `pa` degrees clockwise. If no `location` is given, will assume the center of the frame. `degree` is the corresponding `Interpolations.Degree` for the B-Spline used to subsample the pixel values.

### Coordinate System
* `x, y` - Parsed as distance from the bottom-left corner of the image. Pixel convention is that `(1, 1)` is the center of the bottom-left pixel increasing right and up.
* `r, theta` or `r, θ` - Parsed as polar coordinates from the center of the image. `theta` is a position angle.
"""
function construct(kernel::AbstractMatrix,
                   idxs::Tuple{<:AbstractVector, <:AbstractVector};
                   A=1,
                   pa=0,
                   degree=Linear(),
                   location...)
    T = length(location) > 0 ? mapreduce(typeof, promote_type, values(location)) : eltype(A)
    S = float(T)
    ctr = _frame_center(idxs)
    # have to do reversing and flip sign because `warp` moves canvas not image
    pos = _get_location(ctr, values(location), pa) |> reverse
    tform = Translation(_frame_center(kernel) - pos)
    return  S.(A .* warp(kernel, tform, idxs, degree, zero(S)))
end

construct(kernel, size::Tuple{<:Integer, <:Integer}; kwargs...) =
    construct(kernel, Base.OneTo.(size); kwargs...)

function _frame_center(axes)
    map(axes) do axis
        l, u = extrema(axis)
        (u - l) / 2 + 1
    end |> reverse |> SVector
end

_frame_center(img::AbstractMatrix) = ImageTransformations.center(img) |> reverse

# cartesian
function _get_location(ctr, pars::NamedTuple{(:x, :y)}, pa=0)
    pos = SVector(pars.x, pars.y)
    return recenter(RotMatrix{2}(deg2rad(-pa)), ctr)(pos)
end

_get_location(ctr, pars::NamedTuple{(:y, :x)}, pa=0) = _get_location(ctr, (x=pars.x, y=pars.y), pa)

# polar
function _get_location(ctr, pars::NamedTuple{(:r, :theta)}, pa=0)
    pos = Polar(promote(pars.r, deg2rad(pars.theta - pa))...)
    return CartesianFromPolar()(pos) + ctr
end

_get_location(ctr, pars::NamedTuple{(:theta, :r)}, pa=0) = _get_location(ctr, (r=pars.r, theta=pars.theta), pa)
_get_location(ctr, pars::NamedTuple{(:r, :θ)}, pa=0) = _get_location(ctr, (r=pars.r, theta=pars.θ), pa)
_get_location(ctr, pars::NamedTuple{(:θ, :r)}, pa=0) = _get_location(ctr, (r=pars.r, theta=pars.θ), pa)

# default is just the center
_get_location(ctr, pars::NamedTuple{()}, pa=0) = ctr


################################

"""
    inject(frame, ::PSFKernel; A=1, location...)
    inject(frame, ::AbstractMatrix; A=1, degree=Linear(), location...)

Injects the given PSF kernel or image into `frame` with amplitude `A`. The location can be given in image or polar coordinates, following the coordinate convention below. If no `location` is given, will assume the center of the frame. For empirical PSFs, `degree` is the corresponding `Interpolations.Degree` for the B-Spline used to subsample the pixel values.

### Coordinate System
* `x, y` - Parsed as distance from the bottom-left corner of the image. Pixel convention is that `(1, 1)` is the center of the bottom-left pixel increasing right and up.
* `r, theta` or `r, θ` - Parsed as polar coordinates from the center of the image. `theta` is a position angle.

!!! note
    Due to the integral nature of array indices, frames or images with even-sized axes will have their center rounded to the nearest integer. This may cause unexpected results for small frames and images.

# Examples
```jldoctest
julia> inject(zeros(5, 5), ones(1, 1), A=2, x=2, y=1) # image coordinates
5×5 Array{Float64,2}:
 0.0  2.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> inject(zeros(5, 5), ones(3, 3), r=1.5, theta=90) # polar coords
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  0.0
```
"""
inject(frame::AbstractMatrix, kernel; kwargs...) = inject!(deepcopy(frame), kernel; kwargs...)

"""
    inject!(frame, ::PSFKernel; A=1, location...)
    inject!(frame, ::AbstractMatrix; A=1, degree=Linear(), location...)

In-place version of [`inject`](@ref) which modifies `frame`.
"""
function inject!(frame::AbstractMatrix, kernel; kwargs...)
    return frame .+= construct(kernel, axes(frame); kwargs...)
end

"""
    inject(cube, ::PSFKernel, [angles]; A=1, location...)
    inject(cube, ::AbstractMatrix, [angles]; A=1, degree=Linear(), location...)

Injects `A * img` into each frame of `cube` at the position given by the keyword arguments. If `angles` are provided, the position in the keyword arguments will correspond to the `img` position on the first frame of the cube, with each subsequent repositioned `img` being rotated by `-angles` in degrees. This is useful for fake companion injection. If no `location` is given, will assume the center of each frame. For empirical PSFs, `degree` is the corresponding `Interpolations.Degree` for the B-Spline used to subsample the pixel values.
"""
inject(cube::AbstractArray{T,3}, kernel; kwargs...) where T =
    inject!(deepcopy(cube), kernel; kwargs...)
inject(cube::AbstractArray{T,3}, kernel, angles; kwargs...) where T =
    inject!(deepcopy(cube), kernel, angles; kwargs...)

"""
    inject!(cube, ::PSFKernel, [angles]; A=1, location...)
    inject!(cube, ::AbstractMatrix, [angles]; A=1, location...)

In-place version of [`inject`](@ref) which modifies `cube`.
"""
function inject!(cube::AbstractArray{T,3}, kernel; kwargs...) where T
    for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        inject!(frame, kernel; kwargs...)
    end
    return cube
end

function inject!(cube::AbstractArray{T,3}, kernel, angles::AbstractVector; kwargs...) where T
    size(cube, 1) == length(angles) ||
        error("Number of ADI frames does not much between cube and angles- got $(size(cube, 1)) and $(length(angles))")
    for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        inject!(frame, kernel; pa=angles[idx], kwargs...)
    end
    return cube
end

###################

function normalize_psf!(psf::AbstractMatrix, fwhm, factor=1)
    ap = CircularAperture(center(psf), factor * fwhm / 2)
    area = photometry(ap, psf).aperture_sum
    return psf ./= area
end
normalize_psf(psf, fwhm, factor=1) = normalize_psf!(deepcopy(psf), fwhm, factor)
