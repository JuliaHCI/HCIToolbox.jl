using Statistics
using Photometry.Aperture
using Rotations
using CoordinateTransformations
using ImageTransformations
using Interpolations
using StaticArrays

export Kernels, construct, normalize_psf, normalize_psf!

"""
    Kernels

This module is a simple wrapper around some synthetic PSF kernels.

Available kernels
- [`Kernels.Gaussian`](@ref)/[`Kernels.Normal`](@ref)
- [`Kernels.Moffat`](@ref)
- [`Kernels.Airy`](@ref)
"""
module Kernels

using SpecialFunctions

"""
    Kernels.PSFKernel <: Function

A kernel used for defining a PSF.

# Interface

Right now, each kernel has one function that returns the amplitude of the kernel given a squared Euclidean distance. The amplitude should be scaled appropriately such that the maximum value is 1.
```julia
(kernel::PSFKernel)(sqdist::Number)::Number
```
"""
abstract type PSFKernel <: Function end

"""
    Kernels.Gaussian(fwhm)
    Kernels.Normal(fwhm)

A Gaussian kernel

``K(d) = \\exp\\left[-4\\ln{2}\\left(\\frac{d}{\\Gamma}\\right)^2\\right]``

```
  ┌────────────────────────────────────────┐ 
1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠊⡏⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⡇⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⡇⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠃⠀⠀⡇⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⡇⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⡇⠀⠀⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⡇⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀⠀⠀⠀⡇⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠁⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠘⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
0 │⣀⣀⣀⣀⣀⣀⣀⣀⠤⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⢄⣀⣀⣀⣀⣀⣀⣀│ 
  └────────────────────────────────────────┘ 
  -2fwhm                               2fwhm
```
"""
struct Gaussian{T} <: PSFKernel
    fwhm::T
end

(g::Gaussian)(sqdist) = exp(-4log(2) * sqdist / g.fwhm^2)

# Alias Normal -> Gaussian
const Normal = Gaussian
@doc (@doc Gaussian) Normal


"""
    Kernels.Moffat(fwhm)

A Moffat kernel

``K(d) = \\left[1 + \\left(\\frac{d}{\\Gamma/2}\\right)^2 \\right]^{-1}``

```
  ┌────────────────────────────────────────┐ 
1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⡧⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⡇⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠁⠀⡇⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⡇⠀⢸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⡇⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⡇⠀⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀⠀⠀⠀⡇⠀⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀⠀⠀⠀⡇⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡎⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⢱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⢣⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡜⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠑⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⢀⡠⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠢⣀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⢀⣀⡠⠤⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠒⠤⠤⣀⣀⠀│ 
0 │⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉│ 
  └────────────────────────────────────────┘  
  -2fwhm                               2fwhm
```
"""
struct Moffat{T} <: PSFKernel
    fwhm::T
end

function (m::Moffat)(sqdist)
    hwhm = m.fwhm / 2
    return inv(1 + sqdist / hwhm^2)
end

"""
    Kernels.AiryDisk(fwhm)

An Airy disk. Guaranteed to work even at `r=0`.

``K(r) = \\left[2\\frac{J_1\\left(\\pi r \\right)}{\\pi r}\\right]^2`` where ``r \\approx \\frac{d}{0.97\\Gamma}``

```
  ┌────────────────────────────────────────┐ 
1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠊⡏⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⡇⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⡇⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⡇⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⡇⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠃⠀⠀⠀⡇⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⡇⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⡇⠀⠀⠀⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
0 │⣀⣀⣀⣀⠤⣀⣀⣀⣀⣀⠎⠁⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠣⢄⣀⣀⣀⣀⡠⢄⣀⣀⣀│ 
  └────────────────────────────────────────┘ 
  -2fwhm                               2fwhm
```
"""
struct AiryDisk{T} <: PSFKernel
    fwhm::T
end

const rz = 3.8317059702075125 / π

function (a::AiryDisk)(sqdist)
    radius = a.fwhm * 1.18677
    r = sqrt(sqdist) / (radius / rz)
    return iszero(r) ? 1.0 : (2besselj1(π * r) / (π * r))^2
end

end # module Kernels

using .Kernels: PSFKernel

###################

"""
    construct(::PSFKernel, size; A=1, pa=0, location...)
    construct(::PSFKernel, indices; A=1, pa=0, location...)

Constructs a synthetic PSF using the given kernel.

If `indices` are provided they will be used as the input grid. If `size` is provided the PSF will have the given size. It will have amplitude `A` and will be located at the given `location` (explained below) potentially rotated by an additional `pa` degrees clockwise.

### Coordinate System
* `x, y` - Parsed as distance from the bottom-left corner of the image. Pixel convention is that `(1, 1)` is the center of the bottom-left pixel increasing right and up.
* `r, theta` or `r, θ` - Parsed as polar coordinates from the center of the image. `theta` is a position angle.
"""
function construct(kernel::PSFKernel, idxs::Tuple{<:AbstractVector, <:AbstractVector}; A=1, pa=0, location...)
    S = mapreduce(typeof, promote_type, values(location)) |> float
    ys, xs = idxs
    ctr = _frame_center(idxs)
    x0, y0 = _get_location(ctr, values(location), pa)
    r2 = @. (xs' - x0)^2 + (ys - y0)^2
    return @. S(A * kernel(r2))
end

"""
    construct(::AbstractMatrix, size; A=1, pa=0, location...)
    construct(::AbstractMatrix, indices; A=1, pa=0, location...)

Constructs a PSF using the given matrix via bilinear interpolation.

If `indices` are provided they will be used as the input grid. If `size` is provided the PSF will have the given size. It will have amplitude `A` and will be located at the given `location` (explained below) potentially rotated by an additional `pa` degrees clockwise.

### Coordinate System
* `x, y` - Parsed as distance from the bottom-left corner of the image. Pixel convention is that `(1, 1)` is the center of the bottom-left pixel increasing right and up.
* `r, theta` or `r, θ` - Parsed as polar coordinates from the center of the image. `theta` is a position angle.
"""
function construct(kernel::AbstractMatrix, idxs::Tuple{<:AbstractVector, <:AbstractVector}; A=1, pa=0, location...)
    S = mapreduce(typeof, promote_type, values(location)) |> float
    ctr = _frame_center(idxs)
    # have to do reversing and flip sign because `warp` moves canvas not image
    pos = _get_location(ctr, values(location), pa) |> reverse
    tform = Translation(_frame_center(kernel) - pos)
    return @. S(A * warp(kernel, tform, idxs, Linear(), zero(S)))
end

construct(kernel::PSFKernel, size::Tuple{<:Integer, <:Integer}; A=1, pa=0, location...) =
    construct(kernel, Base.OneTo.(size); A=A, pa=pa, location...)

function _frame_center(axes)
    map(axes) do axis
        l, u = extrema(axis)
        (u - l) / 2 + 1
    end |> reverse |> SVector
end

_frame_center(img::AbstractMatrix) = ImageTransformations.center(img) |> reverse


function _get_location(ctr, pars::NamedTuple{(:x, :y)}, pa=0)
    pos = SVector(pars.x, pars.y)
    return recenter(RotMatrix{2}(deg2rad(-pa)), ctr)(pos)
end

_get_location(ctr, pars::NamedTuple{(:y, :x)}, pa=0) = _get_location(ctr, (x=pars.x, y=pars.y), pa)

function _get_location(ctr, pars::NamedTuple{(:r, :theta)}, pa=0)
    pos = Polar(promote(pars.r, deg2rad(pars.theta - pa))...)
    return CartesianFromPolar()(pos) + ctr
end

_get_location(ctr, pars::NamedTuple{(:theta, :r)}, pa=0) =
    _get_location(ctr, (r=pars.r, theta=pars.theta), pa)
_get_location(ctr, pars::NamedTuple{(:r, :θ)}, pa=0) =
    _get_location(ctr, (r=pars.r, theta=pars.θ), pa)
_get_location(ctr, pars::NamedTuple{(:θ, :r)}, pa=0) =
    _get_location(ctr, (r=pars.r, theta=pars.θ), pa)


################################

"""
    inject(frame, ::PSFKernel; A=1, location...)
    inject(frame, ::AbstractMatrix; A=1, location...)

Injects the given PSF kernel or image into `frame` with amplitude `A`. The location can be given in image or polar coordinates, following the [`Coordinate System`](@ref) convention.

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
 0.0  0.0  1.0  1.0  0.0
 0.0  0.0  1.0  1.0  0.0
```
"""
inject(frame::AbstractMatrix, kernel; A=1, location...) =
    inject!(deepcopy(frame), kernel; A=A, location...)

"""
    inject!(frame, ::PSFKernel; A=1, location...)
    inject!(frame, ::AbstractMatrix; A=1, location...)

In-place version of [`inject`](@ref) which modifies `frame`.
"""
function inject!(frame::AbstractMatrix, kernel; A=1, pa=0, location...)
    return frame .+= construct(kernel, axes(frame); A=A, pa=pa, location...)
end

"""
    inject(cube, ::PSFKernel, [angles]; A=1, location...)
    inject(cube, ::AbstractMatrix, [angles]; A=1, location...)

Injects `A * img` into each frame of `cube` at the position given by the keyword arguments. If `angles` are provided, the position in the keyword arguments will correspond to the `img` position on the first frame of the cube, with each subsequent repositioned `img` being rotated by `-angles` in degrees. This is useful for fake companion injection.
"""
inject(cube::AbstractArray{T,3}, kernel; A=1, location...) where T =
    inject!(deepcopy(cube), kernel; A=A, location...)
inject(cube::AbstractArray{T,3}, kernel, angles; A=1, location...) where T =
    inject!(deepcopy(cube), kernel, angles; A=A, location...)

"""
    inject!(cube, ::PSFKernel, [angles]; A=1, location...)
    inject!(cube, ::AbstractMatrix, [angles]; A=1, location...)

In-place version of [`inject`](@ref) which modifies `cube`.
"""
function inject!(cube::AbstractArray{T,3}, kernel; A=1, location...) where T
    for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        inject!(frame, kernel; A=A, location...)
    end
    return cube
end

function inject!(cube::AbstractArray{T,3}, kernel, angles::AbstractVector; A=1, location...) where T
    size(cube, 1) == length(angles) ||
        error("Number of ADI frames does not much between cube and angles- got $(size(cube, 1)) and $(length(angles))")
    for idx in axes(cube, 1)
        frame = @view cube[idx, :, :]
        inject!(frame, kernel; A=A, pa=angles[idx], location...)
    end
    return cube
end

###################

function normalize_psf!(psf::AbstractMatrix, fwhm, factor=1)
    ap = CircularAperture(center(psf), factor * fwhm / 2)
    area = aperture_photometry(ap, psf).aperture_sum
    return psf ./= area
end
normalize_psf(psf, fwhm, factor=1) = normalize_psf!(deepcopy(psf), fwhm, factor)
