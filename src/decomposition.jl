import MultivariateStats
using Statistics

const mvs = MultivariateStats

export PCA,
       design,
       reduce


abstract type Design end


"""
    PCA

Use principal component analysis (PCA) to reduce data cube. `ncomponents` defines how many principal components to use. 

Uses `MultivariateStats.PCA` for decomposition.

# Returns
Returns a `NamedTuple` with
* `A` - The design Matrix
* `w` - The weight vector fit to our data
* `S` - The reconstruction of the input cube from `A * w`

# Examples

```jldoctest
julia> cube = DataCube(ones(100, 100, 30));

julia> design(PCA, cube)
(A = [1.0; 0.0; … ; 0.0; 0.0], w = [0.0 0.0 … 0.0 0.0], S = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0; … ; 1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0])

```
"""
struct PCA <: Design end

function design(::Type{<:PCA}, cube::DataCube; kwargs...)
    flat_cube = Matrix(cube)

    pca = mvs.fit(mvs.PCA, flat_cube; mean = 0, kwargs...)

    A = mvs.projection(pca)
    weights = mvs.transform(pca, flat_cube)
    reconstructed = mvs.reconstruct(pca, weights)
    return (A = A, w = weights, S = reconstructed)
end

struct NMF <: Design 
    ncomponents::Integer
end

struct Pairet{T <: Design} <: Design
    strategy::T
end


function Base.reduce(D::Type{<:PCA}, cube::DataCube; kwargs...)
    pca = design(D, cube; kwargs...)

    flat_residuals = Matrix(cube) .- pca.S
    cube_residuals = DataCube(flat_residuals, angles(cube))
    collapsed = median(derotate!(cube_residuals))
    return collapsed
end



function Base.reduce(::Type{Pairet{<:PCA}}, cube::DataCube; pca_kwargs...)
    S = []
    reduced = []

    X = Matrix(cube)

    initial_pca = mvs.fit(mvs.PCA, X; maxoutdim = 1)
    

end


