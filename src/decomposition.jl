import MultivariateStats
using Statistics
using LinearAlgebra: I
using NMF: nnmf

const mvs = MultivariateStats

abstract type HCIAlgorithm end

"""
    design(::Type{<:HCIAlgorithm}, cube, args...; kwargs...)

Create a design matrix and weights from the given [`DataCube`](@ref). The `kwargs` will vary based on the design algorithm.

# Returns
The output of a design matrix will be a named tuple with 3 parameters:
* `A` - The design Matrix
* `w` - The weight vector (the transform of our data cube)
* `S` - The reconstruction of our data cube (usually `A * w`)
"""
design

# ------------------------------------------------------------------------------

"""
    PCA

Use principal component analysis (PCA) to reduce data cube.

Uses [`MultivariateStats.PCA`](https://multivariatestatsjl.readthedocs.io/en/stable/pca.html) for decomposition. See [`MultivariateStats.fit(PCA; ...)`](https://multivariatestatsjl.readthedocs.io/en/stable/pca.html#fit) for keyword arguments

# Arguments
* `ncomps::Int` - The number of components to keep. Cannot be larger than the number of frames in the input cube (default).

# Examples

```jldoctest
julia> cube = ones(30, 100, 100);

julia> design(PCA, cube)
(A = [1.0; 0.0; … ; 0.0; 0.0], w = [0.0 0.0 … 0.0 0.0], S = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0])

julia> design(PCA, cube, 5)
(A = [1.0; 0.0; … ; 0.0; 0.0], w = [0.0 0.0 … 0.0 0.0], S = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0])

```
"""
struct PCA <: HCIAlgorithm end

function design(::Type{<:PCA}, cube::AbstractArray{T,3}, ncomps::Integer = size(cube, 1); kwargs...) where T
    flat_cube = flatten(cube)

    pca = mvs.fit(mvs.PCA, flat_cube; mean = 0, maxoutdim = ncomps)

    A = mvs.projection(pca)
    weights = mvs.transform(pca, flat_cube)
    reconstructed = mvs.reconstruct(pca, weights)
    return (A = A, w = weights, S = reconstructed)
end

# ------------------------------------------------------------------------------

"""
    NMF
"""
struct NMF <: HCIAlgorithm end

function design(::Type{<:NMF}, cube::AbstractArray{T,3}, ncomps::Integer = size(cube, 1); kwargs...) where T
    flat_cube = flatten(cube)

    nmf = nnmf(flat_cube, ncomps; kwargs...)
    nmf.converged || @warn "NMF did not converge, try changing `alg`, `maxiter` or `tol` as keyword wargs."

    A = nmf.W
    weights = nmf.H
    reconstructed = nmf.W * nmf.H
    return (A = A, w = weights, S = reconstructed)
end

# ------------------------------------------------------------------------------

"""
    Pairet{<:Union{PCA, NMF}}
"""
struct Pairet{T <: HCIAlgorithm} <: HCIAlgorithm end


function design(::Type{Pairet{<:D}}, cube::AbstractArray{T,3}, ncomps::Integer = size(cube, 1); pca_kwargs...) where {D <: Union{PCA,NMF},T}
    S = []
    reduced = []
    
    X = flatten(cube)

    initial_pca = mvs.fit(mvs.PCA, X; maxoutdim = 1)
    # TODO
end

# ------------------------------------------------------------------------------

"""
    Median

Design using the median of the cube

# Examples
```jldoctest
julia> cube = ones(30, 100, 100);

julia> design(Median, cube)
(A = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], w = UniformScaling{Bool}(true), S = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0])
```

# See Also
[`Mean`](@ref)
"""
struct Median <: HCIAlgorithm end

function design(::Type{<:Median}, cube::AbstractArray{T,3}) where T
    out = median(flatten(cube), dims = 1)
    weights = I
    return (A = out, w = weights, S = out)
end


"""
    Mean

Design using the mean of the cube

# Examples
```jldoctest
julia> cube = DataCube(ones(100, 100, 30));

julia> design(Mean, cube)
(A = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], w = UniformScaling{Bool}(true), S = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0])
```

# See Also
[`Median`](@ref)
"""
struct Mean <: HCIAlgorithm end

function design(::Type{<:Mean}, cube::AbstractArray{T,3}) where T
    out = mean(flatten(cube), dims = 1)
    weights = I
    return (A = out, w = weights, S = out)
end


# ------------------------------------------------------------------------------

"""
    reduce(::Type{<:HCIAlgorithm}, cube, angles, args...; method=median, kwargs...)

Using a given `HCIAlgorithm`, will reduce the cube by first finding the approximate reconstruction with [`design`](@ref) and then derotating and combining (using whichever function specified by `method`). Any `kwargs` will be passed to [`design`](@ref).
"""
function Base.reduce(D::Type{<:HCIAlgorithm},
    cube::AbstractArray{T,3},
    angles::AbstractVector,
    args...;
    method = median,
    kwargs...) where T
    des = design(D, cube, args...; kwargs...)

    flat_residuals = flatten(cube) .- des.S
    cube_residuals = expand(flat_residuals)
    return combine(derotate!(cube_residuals, angles), method = method)
end
