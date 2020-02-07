import MultivariateStats
using Statistics
using LinearAlgebra: I
using NMF: nnmf

const mvs = MultivariateStats

export PCA,
       NMF,
       Pairet,
       Median,
       Mean,
       design,
       reduce

abstract type Design end

"""
    design(::Type{<:Design}, ::DataCube, args...; kwargs...)

Create a design matrix and weights from the given [`DataCube`](@ref). The `kwargs` will vary based on the design algorithm. 

# Returns
The output of a design matrix will be a named tuple with 3 parameters:
* `A` - The design Matrix
* `w` - The weight vector (the transform of our data cube)
* `S` - The reconstruction of our data cube (usually `A * w`)
"""
design(::Type{<:Design}, ::DataCube)

# ------------------------------------------------------------------------------

"""
    PCA

Use principal component analysis (PCA) to reduce data cube. `ncomponents` defines how many principal components to use. 

Uses [`MultivariateStats.PCA`](https://multivariatestatsjl.readthedocs.io/en/stable/pca.html) for decomposition. See [`MultivariateStats.fit(PCA; ...)`](https://multivariatestatsjl.readthedocs.io/en/stable/pca.html#fit) for keyword arguments

# Arguments
* `ncomps::Int = nframes(::DataCube)` - The number of components to keep. Cannot be larger than the number of frames in the DataCube.=

# Examples

```jldoctest
julia> cube = DataCube(ones(100, 100, 30));

julia> design(PCA, cube)
(A = [1.0; 0.0; … ; 0.0; 0.0], w = [0.0 0.0 … 0.0 0.0], S = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0])

```
"""
struct PCA <: Design end

function design(::Type{<:PCA}, cube::DataCube, ncomps::Integer = nframes(cube); kwargs...)
    flat_cube = Matrix(cube)

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
struct NMF <: Design end

function design(::Type{<:NMF}, cube::DataCube, ncomps::Integer = nframes(cube); kwargs...)
    flat_cube = Matrix(cube)

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
struct Pairet{T <: Design} <: Design end


function design(::Type{Pairet{<:D}}, cube::DataCube, ncomps::Integer = nframes(cube); pca_kwargs...) where {D <: Union{PCA,NMF}}
    S = []
    reduced = []

    X = Matrix(cube)

    initial_pca = mvs.fit(mvs.PCA, X; maxoutdim = 1)
end

# ------------------------------------------------------------------------------

"""
    Median

Design using the median of the cube

# Examples
```jldoctest
julia> cube = DataCube(ones(100, 100, 30));

julia> design(Median, cube)
(A = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], w = UniformScaling{Bool}(true), S = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0])
```

# See Also
[`Mean`](@ref)
"""
struct Median <: Design end

function design(::Type{<:Median}, cube::DataCube) 
    A = S = median(cube)
    weights = I
    return (A = A, w = weights, S = S)
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
struct Mean <: Design end
    
function design(::Type{<:Mean}, cube::DataCube) 
    A = S = mean(cube)
    weights = I
    return (A = A, w = weights, S = S)
end


# ------------------------------------------------------------------------------

"""
    reduce(::Type{<:Design}, ::DataCube, args...; collapse=median, kwargs...)

Using a given `Design`, will reduce the [`DataCube`](@ref) by first finding the approximate reconstruction with [`design`](@ref) and then derotating and collapsing (using whichever function specified by `collapse`). Any `kwargs` will be passed to [`design`](@ref).
"""
function Base.reduce(D::Type{<:Design}, cube::DataCube, args...; collapse = median, kwargs...)
    des = design(D, cube, args...; kwargs...)

    flat_residuals = Matrix(cube) .- des.S
    cube_residuals = DataCube(flat_residuals, angles(cube))
    collapsed = collapse(derotate!(cube_residuals))
    return collapsed
end
